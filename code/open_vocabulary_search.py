import os
import glob
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.functional import normalize
import clip
import torch
from torch.functional import F
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.nn.functional import normalize
import seaborn as sns
import matplotlib.pyplot as plt

CLIP_D = 512
TORCH_DEVICE = 'cuda'  # cpu will likely not work

class LSegDir(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return len(sorted(glob.glob("*.pt", root_dir=self.root_dir)))

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, f"frame{idx:05}0.pt")
        x = torch.load(path, map_location=torch.device(TORCH_DEVICE))
        return normalize(x.permute(1, 2, 0), dim=-1)


class OmniDir(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return len(sorted(glob.glob(f"{self.root_dir}/*.npy")))

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, f"feature_{idx:05}.npy")
        x = np.load(path).astype(dtype=np.float16)
        return x.transpose(1, 2, 0)[::5,::5,:]


def prompt_engineering(target_classes):
  return [f"a {t_class} in a scene" for t_class in target_classes]


def cluster(clusterer, features):
    cluster_labels = np.array(clusterer.fit_predict(features))
    unique_clusters = np.unique(cluster_labels)
    cluster_means = []
    for cluster_id in unique_clusters:
        if cluster_id == -1: # Skip the noise cluster
            continue
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_data = features[indices]
        mean_value = np.mean(cluster_data, axis=0)
        cluster_means.append(mean_value)
    return cluster_means


@dataclass
class OVSearchConfig:
    lseg_threshold: float
    lseg_min_cluster_size: int
    min_number_of_views: int

    def __init__(self):
        self.lseg_threshold = 0.9
        self.lseg_min_cluster_size = 100
        self.min_number_of_views = 5
        self.prompt = ""


class OVSearch():
    def __init__(self) -> None:
        lseg_dir = LSegDir("D:\\adl4cv\\replica_room_0\\replica_room_0\\results\\lseg_replica_downscaled")
        omni_dir = OmniDir("D:\\adl4cv\\replica_room_0\\replica_room_0\\results\\objects_feature16")
        self.lseg_loader = DataLoader(lseg_dir, batch_size=None)
        self.omni_loader = DataLoader(omni_dir, batch_size=None)
        self.clip_model, _ = clip.load("ViT-B/32", device=TORCH_DEVICE)

    def process_text(self, text_query):
        text_inputs = clip.tokenize(text_query).cuda()
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
        return F.normalize(text_features, dim=-1)
    
    def compute_clusters_from_lseg_results(self, target_classes, lseg_sim_threshold=0.9, omni_min_cluster_size=50):
        # Text processing
        target_classes_lseg = prompt_engineering(target_classes)
        text_features = self.process_text(target_classes_lseg)
        # Results
        cluster_means = []
        clusterer = HDBSCAN(min_cluster_size=omni_min_cluster_size)

        pbar = tqdm(zip(self.lseg_loader, self.omni_loader), total=len(self.lseg_loader))
        for lseg, omni in pbar:
            # LSeg similarity scores
            lseg_H = lseg.shape[0]
            lseg_W = lseg.shape[1]
            lseg_sim = lseg.reshape(lseg_H * lseg_W, CLIP_D) @ text_features.T
            # Debug
            #sns.heatmap(lseg_sim.cpu().detach().numpy().reshape(lseg_H, lseg_W))
            #plt.show()
            #return
            #assert lseg_sim.amax().item() <= 1.0
            lseg_mask_i = lseg_sim.reshape(lseg_H, lseg_W, lseg_sim.size(-1)).amax(dim=-1).cpu()
            lseg_mask = lseg_mask_i >= lseg_sim_threshold
            if lseg_mask.count_nonzero() <= omni_min_cluster_size:
                continue

            # OmniSeg features which correspond to LSeg mask
            omni_mask = omni[lseg_mask]
            X = normalize(omni_mask.cuda(), dim=-1).cpu().detach().numpy()
            cur_cl_means = cluster(clusterer, X)
            cluster_means.extend(cur_cl_means)
            #break
        return cluster_means

    def ov_search(self, request: OVSearchConfig):
        cluster_means = self.compute_clusters_from_lseg_results(
            [request.prompt],
            lseg_sim_threshold=request.lseg_threshold,
            omni_min_cluster_size=request.lseg_min_cluster_size
        )
        if len(cluster_means) == 0:
            return np.array([])
        clusterer = HDBSCAN(min_cluster_size=request.min_number_of_views)
        final_cluster_means = cluster(clusterer, np.array(cluster_means))
        return np.array(final_cluster_means)
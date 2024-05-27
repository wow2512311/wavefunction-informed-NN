from dgl.data import DGLDataset
import functools
from cif_to_crystal_graph_jarvis import read_cif_jarvis_wavemap, read_cif_jarvis
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader

def get_dataloader(dataset,indices,batch_size,
                   num_workers=0, pin_memory=False):
    sampler = SubsetRandomSampler(indices)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size,
                                 sampler=sampler,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)
    return dataloader

class GraphDataset(DGLDataset):
    def __init__(self,
                 cif_dir,
                 strus,
                 CB_descriptor,
                 VB_descriptor,
                 distance):
        super(GraphDataset, self).__init__(name="GNR_heterojunction_dataset")

        self.cif_path = []
        self.cif_name = []
        self.energygap_data = []
        for stru in strus:
            self.cif_path.append(cif_dir + stru + '.cif')
            self.cif_name.append(stru)
        self.distance = distance
        self.CB_descriptor = CB_descriptor
        self.VB_descriptor = VB_descriptor
        
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_file = self.cif_path[idx]
        #graph = read_cif_jarvis_wavemap(cif_file)
        graph, total_map = read_cif_jarvis_wavemap(cif_file, 
                                                   self.CB_descriptor, 
                                                   self.VB_descriptor,
                                                   self.distance)
        stru_name = self.cif_name[idx]
        return graph, total_map, stru_name, cif_file, graph.num_nodes()

    def __len__(self):
        return len(self.cif_data)
        
    def process(self):
        pass

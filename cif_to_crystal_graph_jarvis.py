import torch 
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import numpy as np
from pymatgen.core.structure import Structure
device = 'cuda:0'

def read_cif_jarvis(cif_file):
    crystal = Structure.from_file(cif_file)
    box = crystal.lattice
    elements = crystal.species
    species_temp = []
    for i,j in enumerate(elements):
        species_temp.append(str(elements[i]))
    coor = crystal.cart_coords
    frac_coor = crystal.frac_coords
    
    boxs = [[box.a,0,0],[0,box.b,0],[0,0,box.c]]
    structure = Atoms(lattice_mat=boxs, coords=coor, elements=species_temp,cartesian=True)
    
    graph = Graph.atom_dgl_multigraph(structure, compute_line_graph = False)
    graph = graph.to(device)
    
    graph.ndata["frac_coords"] = torch.Tensor(np.array(frac_coor)).to(device)
    graph.ndata["frac_coords"].requires_grad = False
    #print(graph.ndata["frac_coords"].requires_grad)
    graph.ndata['atomic_numbers'] = torch.Tensor(np.array(structure.atomic_numbers)).to(device)
    graph.ndata['atomic_numbers'].requires_grad = False
    return graph

def read_cif_jarvis_wavemap(cif_file, CB_descriptor, VB_descriptor, distance):
    crystal = Structure.from_file(cif_file)
    box = crystal.lattice
    elements = crystal.species
    species_temp = []
    for i,j in enumerate(elements):
        species_temp.append(str(elements[i]))
    coor = crystal.cart_coords
    frac_coor = crystal.frac_coords
    
    boxs = [[box.a,0,0],[0,box.b,0],[0,0,box.c]]
    structure = Atoms(lattice_mat=boxs, coords=coor, elements=species_temp,cartesian=True)
    
    graph = Graph.atom_dgl_multigraph(structure, compute_line_graph = False)
    graph = graph.to(device)
    
    graph.ndata["frac_coords"] = torch.Tensor(np.array(frac_coor)).to(device)
    graph.ndata["frac_coords"].requires_grad = False
    #print(graph.ndata["frac_coords"].requires_grad)
    graph.ndata['atomic_numbers'] = torch.Tensor(np.array(structure.atomic_numbers)).to(device)
    graph.ndata['atomic_numbers'].requires_grad = False
    
    cb_logits = CB_descriptor(graph)
    vb_logits = VB_descriptor(graph)
    cb_map = process_wavefunction(graph, cb_logits, distance).unsqueeze(0)
    vb_map = process_wavefunction(graph, vb_logits, distance).unsqueeze(0)
    
    total_map = torch.cat((cb_map,vb_map),dim=0)
    total_map.requires_grad = False    
    return graph, total_map

def process_wavefunction(graph, logits, distance):
    atoms_valence = graph.ndata['atomic_numbers']
    atoms_valence = (atoms_valence-2).clamp(1,10).unsqueeze(1).unsqueeze(1)
    
    frac_coords = graph.ndata['frac_coords']
    frac_coords = frac_coords[:,1:].unsqueeze(1).unsqueeze(1)
    
    temp = torch.exp(-(frac_coords-distance) ** 2)
    temp = temp / temp.sum()
    temp = temp.sum(axis=3)
    
    logits = logits * temp * atoms_valence
    
    logits = logits.sum(axis=0)
    logits = logits / logits.max()

    return logits.reshape(100,100)
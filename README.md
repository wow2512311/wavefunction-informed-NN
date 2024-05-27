# **wavefunction-informed-NN**
 Wavefunction-informed Neural Network for Low-dimensional Interface

# **Repository Structure**
 cif_to_crystal_graph_jarvis.py creates crystal graphs.

 graph_dataset.py creates a dataset based on the generated graph.

 model.py contains four models used in the paper.

 train.py controls the training process. (will be uploaded when the paper is accepted.)

 dataset contains all the data sets (the lumohomo_gap.csv and fermi_level.csv are the files of target properties.) Crystallographic Information File (CIF) is a computer file ending with ".cif", which contains information such as cell parameters, atomic coordinates, literature and so on.

 all trained basic GNN modules (in base) and wavefunction-informed models (in WIF) are uploaded.

 The model of wave function depiction network needs to be trained on the code of repository ** depict-wavefunctions **. https://github.com/wow2512311/depict-wavefunctions

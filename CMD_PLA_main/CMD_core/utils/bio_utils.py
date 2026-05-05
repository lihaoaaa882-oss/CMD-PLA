import numpy as np
from mendeleev import element
from Bio.PDB import PDBParser, Polypeptide

parser = PDBParser(QUIET=True)


ATOM_TYPE = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]
# ATOM_MASS = [element(a).atomic_weight for a in ATOM_TYPE]
NUM_ATOM_TYPE = len(ATOM_TYPE)

BIO_ATOM_TYPE = ["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", 'Si', "P", "S", "Cl",
                 "K", "Ca", "Mn", "Fe", "Co", "Cu", "Zn", "As", "Se", "Br", "I", "Xe", "Au", "Hg"]
NUM_BIO_ATOM_TYPE = len(BIO_ATOM_TYPE)

ATOM2BIO_MAP = [BIO_ATOM_TYPE.index(atom) if atom in BIO_ATOM_TYPE else -1 for atom in ATOM_TYPE]

RES_TYPE_3 = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'TYR', 'ASP', 'HIS', 'ASN', 'GLU',
              'LYS', 'GLN', 'MET', 'ARG', 'SER', 'THR', 'CYS', 'PRO']
RES_TYPE_1 = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S',
              'T', 'C', 'P']
NUM_RES_TYPE = len(RES_TYPE_3)


def get_atype(top):
    """
    :param top: mdtraj topology
    :return: atom type (index of ATOM_TYPE): (N,)
    """
    atype = []
    for atom in top.atoms:
        atom_index = ATOM_TYPE.index(atom.element.symbol)
        atype.append(atom_index)
    atype = np.array(atype, dtype=np.compat.long)
    return atype


def get_seq(pdb_file):
    sequence = ''
    structure = parser.get_structure('anony', pdb_file)
    for model in structure:
        for chain in model:
            polypeptides = Polypeptide.PPBuilder().build_peptides(chain)
            for poly in polypeptides:
                sequence += poly.get_sequence()
    return sequence


def get_rtype(top):
    """
    :param top: mdtraj topology
    :return: residue type of each atom (index of RES_TYPE_3): (N,)
    """
    rtype = []
    for atom in top.atoms:
        residue_index = RES_TYPE_3.index(atom.residue.name)
        rtype.append(residue_index)
    rtype = np.array(rtype, dtype=np.compat.long)
    return rtype


def get_res_mask(top):
    """
    :param top: mdtraj topology
    :return: residue mask: (N,)
    """
    rmask = [atom.residue.index for atom in top.atoms]
    rmask = np.array(rmask, dtype=np.compat.long)
    return rmask


def get_backbone_index(top):
    """
    :param top: mdtraj topology
    :return: backbone index of each residue, order: (N, CA, C, O), shape: (B, 4)
    """
    bb_index = []
    for residue in top.residues:
        backbone = [residue.atom(atom_name) for atom_name in ['N', 'CA', 'C', 'O'] if
                    residue.atom(atom_name) is not None]
        bb_index.append([atom.index for atom in backbone])
    bb_index = np.array(bb_index, dtype=np.compat.long)
    return bb_index

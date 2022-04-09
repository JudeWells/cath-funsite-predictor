gcf_path = '/Users/judewells/Documents/dataScienceProgramming/alphafold_funsite/sequence_mapping/chaingcf/'

def get_gcf_sequences_and_mapping(pdbid:str,  dir:str=gcf_path, pdb_to_letter=False):
    filepath = dir + pdbid + '.gcf'
    with open(filepath, 'r') as filehandle:
        lines = filehandle.readlines()
    fasta = extract_fasta_sequence(lines)
    pdb = extract_pdb_sequence(lines)
    mapping = extract_mapping(lines)
    if pdb_to_letter:
        pdb_num_to_letter_dict = extract_pdb2letter(lines)
        return fasta, pdb, mapping, pdb_num_to_letter_dict
    return fasta, pdb, mapping


def extract_fasta_sequence(lines):
    return ''.join([l[0] for l in lines[1:]])

def extract_pdb_sequence(lines):
    return ''.join([l[14] for l in lines[1:]])

def extract_pdb2letter(lines):
    pdb2letter = {l[7:12]: l[14] for l in lines[1:]}
    pdb2letter = {k.strip(): v for k,v in pdb2letter.items()}
    return pdb2letter

def extract_mapping(lines):
    mapping =  {l[7:12]: l[1:7] for l in lines[1:]}
    mapping = {k.strip(): v.strip() for k,v in mapping.items()}
    try:
        del mapping['']
    except KeyError:
        pass
    return {k: int(v) for k,v in mapping.items()}
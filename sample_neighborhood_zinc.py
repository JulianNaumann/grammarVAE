import sys
import molecule_vae
import numpy as np
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Explore the latent space of the autoencoder')
    parser.add_argument('-s', '--smile', type=str, metavar='SMILE_STRING', default='C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]', help='SMILE which neighborhood will be explored (if not provided sample smile will be used)')
    parser.add_argument('-d', '--delta', type=float , metavar='DELTA', default=0.1, help='Step size for exploration')
    parser.add_argument('-t', '--size', type=int, metavar='GRIDSIZE', default=11, help='Size of the grid, must be uneven')
    parser.add_argument('-b1', '--base1dim', type=int, metavar='DIM', default=0, help='Dimension of base vector 1 which will be non-zero')
    parser.add_argument('-b2', '--base2dim', type=int, metavar='DIM', default=1, help='Dimension of base vector 2 which will be non-zero')
    return parser.parse_args()


def save_grid(list, base1dim, base2dim, delta, gridsize):
    filename = 'gridsample/smiles_b1-{}_b2-{}_d-{}_t-{}.txt'.format(base1dim, base2dim, delta, gridsize)
    with open(filename, 'w') as f:
        for r in range(len(list)):
            for c in range(len(list[0])):
                f.write(list[r][c])
                if c < len(list[0]) - 1:
                    f.write('\t')
            if r < len(list) - 1:
                f.write('\n')


def get_neighborhood(grammar_model, latent_epicenter, gridsize, basevector1, basevector2):
    smiles = [[]]
    offset = gridsize / 2
    for r in range(gridsize):
        for c in range(gridsize):
            neighbor_candidates = []
            latent_point = latent_epicenter + (r - offset) * basevector1 + (c - offset) * basevector2
            # sample 1000 times and then select element with most occurence
            for i in range(1000):
                sampled_neighbor = grammar_model.decode(latent_point)[0]
                neighbor_candidates.append(sampled_neighbor)
            best_fit = max(neighbor_candidates, key=neighbor_candidates.count)
            smiles[r].append(best_fit)
        if r < gridsize - 1:
            smiles.append([])
    return smiles

def main():
    args = get_arguments()
    grammar_weights = "pretrained/zinc_vae_grammar_L56_E100_val.hdf5"
    grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)

    smile_epicenter = [args.smile]
    latent_epicenter = grammar_model.encode(smile_epicenter)

    basevector1 = np.zeros((1,56), dtype=np.float32)
    basevector1[0][args.base1dim] = np.float32(args.delta)
    basevector2 = np.zeros((1,56), dtype=np.float32)
    basevector2[0][args.base2dim] = np.float32(args.delta)

    smiles = get_neighborhood(grammar_model, latent_epicenter, args.size, basevector1, basevector2)
    save_grid(smiles, args.base1dim, args.base2dim, args.delta, args.size)


if __name__ == '__main__':
    main()

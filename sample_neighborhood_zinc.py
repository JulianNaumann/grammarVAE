import sys
import molecule_vae
import numpy as np
import argparse
import time

def get_arguments():
    parser = argparse.ArgumentParser(description='Explore the latent space of the autoencoder')
    parser.add_argument('-s', '--smile', type=str, metavar='SMILE_STRING', default='C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]', help='SMILE which neighborhood will be explored (if not provided sample smile will be used)')
    parser.add_argument('-t', '--size', type=int, metavar='GRIDSIZE', default=11, help='Size of the grid, must be uneven')
    parser.add_argument('-d', '--delta', type=float , metavar='DELTA', default=0.2, help='Step size for exploration')
    return parser.parse_args()


def save_grid(list, gridsize, delta):
    filename = 'gridsample/smiles_t{}_d{}.txt'.format(gridsize, delta)
    with open(filename, 'w') as f:
        for r in range(len(list)):
            for c in range(len(list[0])):
                f.write(list[r][c])
                if c < len(list[0]) - 1:
                    f.write('\t')
            if r < len(list) - 1:
                f.write('\n')


def get_neighborhood(grammar_model, latent_epicenter, gridsize, unitvector1, unitvector2):
    smiles = []
    offset = gridsize / 2
    now = time.time()
    for r in range(gridsize):
        smiles.append([])
        for c in range(gridsize):
            num_already = r * gridsize + c + 1
            num_total = gridsize**2
            eta = (time.time()-now) / num_already * (num_total - num_already)
            sys.stdout.write('\rSampling molecule {} of {}, ETA: {:.2f}'.format(num_already, num_total, eta))
            sys.stdout.flush()
            
            neighbor_candidates = []
            latent_point = latent_epicenter + (r - offset) * unitvector1 + (c - offset) * unitvector2
            # sample 1000 times and then select element with most occurence
            for i in range(1):
                sampled_neighbor = grammar_model.decode(latent_point)[0]
                neighbor_candidates.append(sampled_neighbor)
            best_fit = max(neighbor_candidates, key=neighbor_candidates.count)
            smiles[r].append(best_fit)
    sys.stdout.write('\n')
    return smiles


def getOrthogonal(normal):
    v = np.random.randn(normal.size)
    dim = np.random.randint(0, normal.size)
    v[dim] = 0.0

    v[dim] = float(np.dot(normal, v)) / normal[dim] * -1
    v = v / np.linalg.norm(v)
    return v


def main():
    args = get_arguments()
    grammar_weights = "pretrained/zinc_vae_grammar_L56_E100_val.hdf5"
    grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)

    latent_epicenter = grammar_model.encode([args.smile])
    unitvector1 = getOrthogonal(latent_epicenter[0])
    unitvector2 = getOrthogonal(latent_epicenter[0])

    unitvector1 = unitvector1 * args.delta
    unitvector2 = unitvector2 * args.delta

    smiles = get_neighborhood(grammar_model, latent_epicenter, args.size, unitvector1, unitvector2)
    save_grid(smiles, args.size, args.delta)


if __name__ == '__main__':
    main()

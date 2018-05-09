import sys, os
import molecule_vae
import numpy as np
import argparse
import time
import rdkit.Chem
import rdkit.Chem.Draw
#import rdkit.rdBase
import rdkit.RDLogger
from PIL import Image

#rdBase.DisableLog('rdApp.*')
logger = rdkit.RDLogger.logger()
logger.setLevel(rdkit.RDLogger.CRITICAL)

def get_arguments():
    parser = argparse.ArgumentParser(description='Explore the latent space of the autoencoder')
    parser.add_argument('-s', '--smile', type=str, metavar='SMILE_STRING', default='C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]', help='SMILE which neighborhood will be explored (if not provided sample smile will be used)')
    parser.add_argument('-t', '--gridsize', type=int, metavar='GRIDSIZE', default=11, help='Size of the grid, must be uneven')
    parser.add_argument('-d', '--delta', type=float , metavar='DELTA', default=0.2, help='Step size for exploration')
    parser.add_argument('-k', '--numberSamples', type=int , metavar='NUMBER_OF_SAMPLES', default=1000, help='How often each grid point is sampled')
    parser.add_argument('-v', '--validateSmiles', action='store_true', help='Output only valid smiles strings')
    return parser.parse_args()


def stich_image(smiles, gridsize, imageSize):
    final_width = final_height = imageSize * gridsize
    final_image = Image.new('L', (final_width, final_height))
    for row in range(gridsize):
        for col in range(gridsize):
            if smiles[row][col] == 'invalid':
                moleculeImage = Image.open("./molecule_neighborhood/invalid_molecule.png")
            else:
                molecule = rdkit.Chem.MolFromSmiles(smiles[row][col])
                moleculeImage = rdkit.Chem.Draw.MolToImage(molecule, size=(imageSize, imageSize))
            moleculeImage = moleculeImage.resize((imageSize, imageSize))
            final_image.paste(moleculeImage, (col * imageSize, row * imageSize))
    return final_image


def getBestFit(neighbor_candidates, validateSmiles):
    isValid = False
    while not isValid:
        bestFit = max(neighbor_candidates, key=neighbor_candidates.count)
        mol = rdkit.Chem.MolFromSmiles(bestFit)
        neighbor_candidates = filter(lambda e: e!=bestFit, neighbor_candidates)
        isValid = (mol is not None) or (not validateSmiles) or (not neighbor_candidates)
    if (mol is None) and validateSmiles:
        bestFit = 'invalid'
    return bestFit


def get_neighborhood(grammar_model, latent_epicenter, gridsize, numberSamples, validateSmiles, unitvector1, unitvector2):
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
            # sample #numberSamples times and then select element with most occurence
            for i in range(numberSamples):
                sampled_neighbor = grammar_model.decode(latent_point)[0]
                neighbor_candidates.append(sampled_neighbor)
            bestFit = getBestFit(neighbor_candidates, validateSmiles)
            smiles[r].append(bestFit)
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

    smiles = get_neighborhood(grammar_model, latent_epicenter, args.gridsize, args.numberSamples, args.validateSmiles, unitvector1, unitvector2)

    IMAGESIZE = 150 # in pixels
    filename = 'molecule_neighborhood/smiles_t{}_d{}_k{}_v{}.png'.format(args.gridsize, args.delta, args.numberSamples, args.validateSmiles)
    final_image = stich_image(smiles, args.gridsize, IMAGESIZE)
    final_image.save(filename)


if __name__ == '__main__':
    main()

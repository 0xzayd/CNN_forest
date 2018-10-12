# Launch this

from src import CNN2DModel, Data
import argparse

parser = argparse.ArgumentParser(description='arguments input')
parser.add_argument('-l','--layers', type=str, help='e.g "64, 32, 256"', required=True)
parser.add_argument('-id', '--id', type=int, help='id', required=True)
parser.add_argument('-lr', '--lrate', type=float, help='learning rate', required=True)

args = parser.parse_args()

layers = [int(item) for item in args.layers.split(',')]

sim_id = args.id

lr = args.lrate

myModel = CNN2DModel(num_gpus = 1, sim_id = sim_id)

myModel.build_model(blocks=layers)

myModel.compile_model(lr = lr, verbose=True)

myModel.build_callbacks('./logs'+str(sim_id))

#witout generators
data = Data()
X_data, Y_data = data.load_XY(pathX = './train/X_data.npz', pathY = './train/Y_data.npz')

#load checkpoint
myModel.load_checkpoint()

myModel.fit_model(X_data,Y_data, epochs=80)
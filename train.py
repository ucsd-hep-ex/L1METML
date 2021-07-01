import argparse
import time
import os

if __name__ == "__main__":
    time_path = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    path = "./result/"+time_path+"_PUPPICandidates/"
    
    # 4 arguments
    # train from h5 or root?
    # root file location
    # output path
    # mode 0 or 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataSetType', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--input', action='store', type=str, required=False, help='designate input file path')
    parser.add_argument('--output', action='store', type=str, required=True, help='designate output file path')
    parser.add_argument('--mode', action='store',   type=int, required=True, help='0 for L1MET, 1 for DeepMET')
    args = parser.parse_args()

dataSetType = args.dataSetType
if dataSetType == 'h5':
    inputPath = args.input
    # convert root files to h5 and store in same location
    i =0
    for file in os.listdir(inputPath):
        if '.root' in file:
                os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {inputPath}/{file} -o {inputPath}')
                i += 1
                os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {inputPath}/{file} -o {inputPath}/set{i}.h5')
    # place h5 file names into a .txt file
    writeFile= open(f'{inputPath}/h5files.txt',"w+")
    for file in os.listdir(inputPath):
        if '.h5' in file:
            writeFile.write(f'{inputPath}/{file}\n')
    writeFile.close
    # this file is read in main()
    h5files = f'{inputPath}/h5files.txt'}
    h5filesList = f'{inputPath}/h5files.txt'
            
    from train_fromh5 import main
    main(args, h5filesList)

if dataSetType == 'root':
    from train_fromRoot import main
    main(args)

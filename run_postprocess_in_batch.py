import os

def run_marine():
    col_names = ["dataset", "simplification level", "alpha", "superlevel set threshold"]
    runs = []
    
    dataset_names = ["CPPin20230801_", "CPPin20230802_", "CPPin20230803_", "CPPin20230804_",
                     "CPPin20230805_", "CPPin20230806_", "CPPin20230807_", "CPPin20230808_"]
    simp_levels = ["0"]
    alphas = ["0.4"]
    super_thresholds = ["2.0"]
    
    for dataset in dataset_names:
        for simplvl in simp_levels:
            for alpha in alphas:
                for sup_thres in super_thresholds:
                    with open("postprocess.config", "w") as outf:
                        print("# {}".format(col_names[0]), file=outf)
                        print(dataset, file=outf)
                        print(file=outf)
                        
                        print("# {}".format(col_names[1]), file=outf)
                        print(simplvl, file=outf)
                        print(file=outf)
                        
                        print("# {}".format(col_names[2]), file=outf)
                        print(alpha, file=outf)
                        print(file=outf)
                        
                        print("# {}".format(col_names[3]), file=outf)
                        print(sup_thres, file=outf)
                    
                        outf.close()
                    
                    os.system("python ./MarineCloud/postprocess.py pFGW-system")
                    
def run_land():
    col_names = ["dataset", "simplification level", "alpha", "superlevel set threshold"]
    runs = []
    
    dataset_names = ["20180501_juelich", "20180623_juelich", "20190512_juelich"]
    alphas = ["0.2"]
    
    for dataset in dataset_names:
        for alpha in alphas:
            with open("postprocess_juelich.config", "w") as outf:
                print("# {}".format(col_names[0]), file=outf)
                print(dataset, file=outf)
                print(file=outf)
                
                print("# {}".format(col_names[1]), file=outf)
                print(file=outf)
                
                print("# {}".format(col_names[2]), file=outf)
                print(alpha, file=outf)
                print(file=outf)
                
                print("# {}".format(col_names[3]), file=outf)
                print("0-36: 9.0", file=outf)
                print("37-108: 10.0", file=outf)
                print("109-: 9.0", file=outf)
            
                outf.close()
            
            os.system("python ./LandCloud/postprocess_juelich.py pFGW-system")

if __name__ == '__main__':
    run_marine()
    run_land()
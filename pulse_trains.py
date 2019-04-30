models = ["rattay01", "briaire05","smit10","imennov09"]
train_data = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model in enumerate(models):
    
    train_data[ii] = pd.read_csv("results/Analyses/train_response_{}.csv".format(model))
    
    train_data[ii]["model"] = model

pulse_train_data = pd.concat(train_data,ignore_index = True)

pulse_train_data["nof_spikes"] = 1

pulse_train_data = pulse_train_data[["model","pulse rate","amplitude","stimulus amplitude (uA)","nof_spikes"]]

pulse_train_data = pulse_train_data.groupby(["model","pulse rate","amplitude","stimulus amplitude (uA)"]).sum()

pulse_train_data.reset_index(inplace=True)

pulse_train_data.to_csv("results/pulse_train_data.csv", index=False, header=True)   

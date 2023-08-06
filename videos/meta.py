from os import listdir

for filename in listdir("UWB/"):
    with open(f'UWB/{filename}', 'r') as uwb:
        uwbData = uwb.readlines()
    xyTime = []
    for x in uwbData:
        g = x.split(",")
        if g[2] == 'nan':
            pass
        if g[3] == 'nan':
            pass
        xyTime.append([g[2], g[3], g[5].replace("\n", "")])
    print(filename)
    print("\n")
    print(xyTime)
    
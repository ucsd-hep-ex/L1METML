import numpy as np
import matplotlib.pyplot as plt
import math


loss_file = open("loss_data.log", 'r')

table = loss_file.readline()
table_split = table.split(",")


print("###############\n")

print("Select variable\n")


for i, var in enumerate(table_split):
    print("{}\t {}".format(i, var))

    if '\n' in var:
        table_split[i] = table_split[i].replace('\n','')


num_var = int(input("Input number 0 ~ 10 : "))

print("\n###############\n")


print("Plot {} vs epoch curve".format(table_split[num_var]))

epoch_ = 0
plot_maxi = 0
array_epoch = []
array_var = []

while True:
    line = loss_file.readline()
    if not line: break

    line_split = line.split(",")
    temp_var = line_split[num_var]
    
    array_epoch.append(epoch_)
    array_var.append(temp_var)

    if plot_maxi < float(temp_var):
        plot_maxi = float(temp_var)

    epoch_ += 1

array_var = list(map(float, array_var))

plot_maxi = plot_maxi * 1.1


plt.plot(array_epoch, array_var)
plt.title(table_split[num_var])
plt.xlabel("epoch")
plt.ylabel(table_split[num_var])
#plt.ylim(0,5000)
#plt.ylim(0,plot_maxi)
#plt.xlim(0,140)
plt.grid()
plt.show()
plt.savefig("loss_plot/{}.png".format(table_split[num_var]))    

loss_file.close()

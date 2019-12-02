from  data_control import Data_intialize
from learn_setup_dqn import Learn_set
from test_dqn import Test_dqn
from network import *
from circuit_plot import create_schem


# #one house model
# net=Network(new=True)
# bus1=net.create_bus(name="bus1")
# bus2=net.create_bus(name="bus2")
# _,h1=net.create_housepvb(bus1,p_w=4560,pv="pv_2kw",storage="battery_4kwh")
# _,h2=net.create_housepvb(bus2,p_w=4560,pv="pv_2kw",storage="battery_4kwh")
# net.create_ext_grid(bus2,max_p_mw=20)
# net.create_line(bus1,bus2,length_km=1)
# #definde group for learning
# print(net.net)
# group1=[h1,h2]
# groups=[group1]
# data_set=Data_intialize(net.net)
# ls=Learn_set(net.net,groups,Data_intialize)
# create_schem(net)

net=Network(new=True)
bus1=net.create_bus(name="bus1")
bus2=net.create_bus(name="bus2")
bus3=net.create_bus(name="bus3")
bus4=net.create_bus(name="bus4")
bus5=net.create_bus(name="bus1")
bus6=net.create_bus(name="bus2")
bus7=net.create_bus(name="bus3")
bus8=net.create_bus(name="bus4")
bus9=net.create_bus(name="bus4")
bus10=net.create_bus(name="bus1")
bus11=net.create_bus(name="bus2")
bus12=net.create_bus(name="bus4")
bus13=net.create_bus(name="bus4")
bus14=net.create_bus(name="bus1")
bus15=net.create_bus(name="bus2")
bus16=net.create_bus(name="bus1")
_,h1=net.create_housepvb(bus2,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h2=net.create_housepvb(bus3,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h3=net.create_housepvb(bus4,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h4=net.create_housepvb(bus5,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h5=net.create_housepvb(bus6,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h6=net.create_housepvb(bus7,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h7=net.create_housepvb(bus8,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h8=net.create_housepvb(bus9,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h9=net.create_housepvb(bus10,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h10=net.create_housepvb(bus11,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h11=net.create_housepvb(bus12,p_w=400,pv="pv_2kw",storage="battery_4kwh",direct='right')
_,h12=net.create_housepvb(bus13,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h13=net.create_housepvb(bus14,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h14=net.create_housepvb(bus15,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h15=net.create_housepvb(bus16,p_w=400,pv="pv_2kw",storage="battery_4kwh")
net.create_ext_grid(bus1,max_p_mw=20)
net.create_line(bus1,bus2,length_km=1)
net.create_line(bus2,bus3,length_km=1)
net.create_line(bus3,bus4,length_km=1)
net.create_line(bus1,bus5,length_km=1,direct="down",theta=-90)
net.create_line(bus5,bus6,length_km=1)
net.create_line(bus6,bus7,length_km=1)
net.create_line(bus7,bus8,length_km=1)
net.create_line(bus5,bus9,length_km=1,direct="down",theta=-90)
net.create_line(bus9,bus10,length_km=1)
net.create_line(bus10,bus11,length_km=1)
net.create_line(bus1,bus12,length_km=1,direct="left",theta=180)
net.create_line(bus12,bus13,length_km=1,direct="left",theta=180)
net.create_line(bus5,bus14,length_km=1,direct="left",theta=180)
net.create_line(bus9,bus15,length_km=1,direct="left",theta=180)
net.create_line(bus15,bus16,length_km=1,direct="left",theta=180)
#create_schem(net)
#definde group for learning
print(net.net)
group1=[h1,h2,h3]
group2=[h4,h5,h6,h7]
group3=[h8,h9,h10]
group4=[h11,h12]
group5=[h13]
group6=[h14,h15]
groups=[group1,group2,group3,group4,group5,group6]
data=Data_intialize(net.net)
#ls=Learn_set(net.net,groups,Data_intialize)
#data=Data_intialize(net.net)
ts= Test_dqn(net.net,groups)
# create_schem(net)

from mathe_opt_setup import Mathe_set
import sys
sys.path.append('./')
from  data_control import Data_intialize
from network import *
from circuit_plot import create_schem

net=Network(new=True)
bus0=net.create_bus(name="bus1")
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
bus17=net.create_bus(name="bus4")
bus18=net.create_bus(name="bus4")
bus19=net.create_bus(name="bus1")
bus20=net.create_bus(name="bus2")
bus21=net.create_bus(name="bus1")
bus22=net.create_bus(name="bus1")
bus23=net.create_bus(name="bus2")
bus24=net.create_bus(name="bus3")
bus25=net.create_bus(name="bus4")
bus26=net.create_bus(name="bus1")
bus27=net.create_bus(name="bus2")
bus28=net.create_bus(name="bus3")
bus29=net.create_bus(name="bus4")
bus30=net.create_bus(name="bus4")
bus31=net.create_bus(name="bus1")
bus32=net.create_bus(name="bus2")
bus33=net.create_bus(name="bus4")
bus34=net.create_bus(name="bus4")
bus35=net.create_bus(name="bus1")
bus36=net.create_bus(name="bus2")
bus37=net.create_bus(name="bus1")
bus38=net.create_bus(name="bus4")
bus39=net.create_bus(name="bus4")
bus40=net.create_bus(name="bus1")
bus41=net.create_bus(name="bus2")
bus42=net.create_bus(name="bus1")
bus43=net.create_bus(name="bus1")
bus44=net.create_bus(name="bus2")
bus45=net.create_bus(name="bus1")
bus46=net.create_bus(name="bus1")
bus47=net.create_bus(name="bus2")
bus48=net.create_bus(name="bus2")
bus49=net.create_bus(name="bus1")
bus50=net.create_bus(name="bus1")
bus51=net.create_bus(name="bus1")
bus52=net.create_bus(name="bus1")
bus53=net.create_bus(name="bus1")
bus54=net.create_bus(name="bus1")
bus55=net.create_bus(name="bus1")
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
_,h11=net.create_housepvb(bus12,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h12=net.create_housepvb(bus13,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h13=net.create_housepvb(bus14,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h14=net.create_housepvb(bus15,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h15=net.create_housepvb(bus16,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h16=net.create_housepvb(bus17,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h17=net.create_housepvb(bus18,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h18=net.create_housepvb(bus19,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h19=net.create_housepvb(bus20,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h20=net.create_housepvb(bus21,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h21=net.create_housepvb(bus22,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h22=net.create_housepvb(bus23,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h23=net.create_housepvb(bus24,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h24=net.create_housepvb(bus25,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h25=net.create_housepvb(bus26,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h26=net.create_housepvb(bus27,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h27=net.create_housepvb(bus28,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h28=net.create_housepvb(bus29,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h29=net.create_housepvb(bus30,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h30=net.create_housepvb(bus31,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h31=net.create_housepvb(bus32,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h32=net.create_housepvb(bus33,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h33=net.create_housepvb(bus34,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h34=net.create_housepvb(bus35,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h35=net.create_housepvb(bus36,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h36=net.create_housepvb(bus37,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h37=net.create_housepvb(bus38,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h38=net.create_housepvb(bus39,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h39=net.create_housepvb(bus40,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h40=net.create_housepvb(bus41,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h41=net.create_housepvb(bus42,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h42=net.create_housepvb(bus43,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h43=net.create_housepvb(bus44,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h44=net.create_housepvb(bus45,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h45=net.create_housepvb(bus46,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h46=net.create_housepvb(bus47,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h47=net.create_housepvb(bus48,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h48=net.create_housepvb(bus49,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h49=net.create_housepvb(bus50,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h50=net.create_housepvb(bus51,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h51=net.create_housepvb(bus52,p_w=400,pv="pv_2kw",storage="battery_4kwh")
net.create_ext_grid(bus1,max_p_mw=20)
net.create_line(bus1,bus2,length_km=1)
net.create_line(bus2,bus3,length_km=1)
net.create_line(bus3,bus4,length_km=1)
net.create_line(bus4,bus5,length_km=1)
net.create_line(bus5,bus6,length_km=1)
net.create_line(bus6,bus7,length_km=1)
net.create_line(bus7,bus8,length_km=1)
net.create_line(bus8,bus9,length_km=1)
net.create_line(bus9,bus10,length_km=1)
net.create_line(bus10,bus11,length_km=1)
net.create_line(bus1,bus12,length_km=1,direct="down",theta=-90)
net.create_line(bus12,bus13,length_km=1)
net.create_line(bus13,bus14,length_km=1)
net.create_line(bus14,bus15,length_km=1)
net.create_line(bus15,bus16,length_km=1)
net.create_line(bus16,bus17,length_km=1)
net.create_line(bus17,bus18,length_km=1)
net.create_line(bus18,bus19,length_km=1)
net.create_line(bus19,bus20,length_km=1)
net.create_line(bus20,bus21,length_km=1)
net.create_line(bus1,bus24,length_km=1,direct="down",theta=180)
net.create_line(bus24,bus25,length_km=1,theta=180)
net.create_line(bus25,bus26,length_km=1,theta=180)
net.create_line(bus26,bus27,length_km=1,theta=180)
net.create_line(bus27,bus28,length_km=1,theta=180)
net.create_line(bus28,bus29,length_km=1,theta=180)
net.create_line(bus29,bus30,length_km=1,theta=180)
net.create_line(bus30,bus31,length_km=1,theta=180)
net.create_line(bus31,bus32,length_km=1,theta=180)
net.create_line(bus32,bus33,length_km=1,theta=180)
net.create_line(bus12,bus22,length_km=1,direct="down",theta=-90)
net.create_line(bus22,bus23,length_km=1,direct="right")
net.create_line(bus23,bus44,length_km=1,direct="right")
net.create_line(bus44,bus45,length_km=1,direct="right")
net.create_line(bus45,bus46,length_km=1,direct="right")
net.create_line(bus46,bus47,length_km=1,direct="right")
net.create_line(bus47,bus48,length_km=1,direct="right")
net.create_line(bus48,bus49,length_km=1,direct="right")
net.create_line(bus49,bus50,length_km=1,direct="right")
net.create_line(bus50,bus51,length_km=1,direct="right")
net.create_line(bus12,bus34,length_km=1,direct="down",theta=180)
net.create_line(bus34,bus35,length_km=1,theta=180)
net.create_line(bus35,bus36,length_km=1,theta=180)
net.create_line(bus36,bus37,length_km=1,theta=180)
net.create_line(bus37,bus38,length_km=1,theta=180)
net.create_line(bus38,bus39,length_km=1,theta=180)
net.create_line(bus39,bus40,length_km=1,theta=180)
net.create_line(bus40,bus41,length_km=1,theta=180)
net.create_line(bus41,bus42,length_km=1,theta=180)
net.create_line(bus42,bus43,length_km=1,theta=180)

#create_schem(net)
group1=[h51,h2,h3,h4,h5,h6,h7,h8,h9,h10]
group2=[h11,h12,h13,h14,h15,h16,h17,h18,h19,h20]
group3=[h21,h22,h23,h24,h25,h26,h27,h28,h29,h30]
group4=[h31,h32,h33,h34,h35,h36,h37,h38,h39,h40]
group5=[h41,h42,h43,h44,h45,h46,h47,h48,h49,h50]

groups=[group1,group2,group3,group4,group5]
data=Data_intialize(net.net)
ls=Mathe_set(net.net,groups,Data_intialize)

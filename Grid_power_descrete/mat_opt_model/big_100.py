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
bus56=net.create_bus(name="bus1")
bus57=net.create_bus(name="bus1")
bus58=net.create_bus(name="bus2")
bus59=net.create_bus(name="bus3")
bus60=net.create_bus(name="bus4")
bus61=net.create_bus(name="bus1")
bus62=net.create_bus(name="bus2")
bus63=net.create_bus(name="bus3")
bus64=net.create_bus(name="bus4")
bus65=net.create_bus(name="bus4")
bus66=net.create_bus(name="bus1")
bus67=net.create_bus(name="bus2")
bus68=net.create_bus(name="bus4")
bus69=net.create_bus(name="bus4")
bus70=net.create_bus(name="bus1")
bus71=net.create_bus(name="bus2")
bus72=net.create_bus(name="bus1")
bus73=net.create_bus(name="bus4")
bus74=net.create_bus(name="bus4")
bus75=net.create_bus(name="bus1")
bus76=net.create_bus(name="bus2")
bus77=net.create_bus(name="bus1")
bus78=net.create_bus(name="bus1")
bus79=net.create_bus(name="bus2")
bus80=net.create_bus(name="bus3")
bus81=net.create_bus(name="bus4")
bus82=net.create_bus(name="bus1")
bus83=net.create_bus(name="bus2")
bus84=net.create_bus(name="bus3")
bus85=net.create_bus(name="bus4")
bus86=net.create_bus(name="bus4")
bus87=net.create_bus(name="bus1")
bus88=net.create_bus(name="bus1")
bus89=net.create_bus(name="bus2")
bus90=net.create_bus(name="bus4")
bus91=net.create_bus(name="bus4")
bus92=net.create_bus(name="bus1")
bus93=net.create_bus(name="bus2")
bus94=net.create_bus(name="bus1")
bus95=net.create_bus(name="bus4")
bus96=net.create_bus(name="bus4")
bus97=net.create_bus(name="bus1")
bus98=net.create_bus(name="bus2")
bus99=net.create_bus(name="bus1")
bus100=net.create_bus(name="bus1")
bus101=net.create_bus(name="bus2")
bus102=net.create_bus(name="bus1")
bus103=net.create_bus(name="bus1")
bus104=net.create_bus(name="bus2")
bus105=net.create_bus(name="bus2")
bus106=net.create_bus(name="bus1")
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
_,h52=net.create_housepvb(bus53,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h53=net.create_housepvb(bus54,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h54=net.create_housepvb(bus55,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h55=net.create_housepvb(bus56,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h56=net.create_housepvb(bus57,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h57=net.create_housepvb(bus58,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h58=net.create_housepvb(bus59,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h59=net.create_housepvb(bus60,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h60=net.create_housepvb(bus61,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h61=net.create_housepvb(bus62,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h62=net.create_housepvb(bus63,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h63=net.create_housepvb(bus64,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h64=net.create_housepvb(bus65,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h65=net.create_housepvb(bus66,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h66=net.create_housepvb(bus67,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h67=net.create_housepvb(bus68,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h68=net.create_housepvb(bus69,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h69=net.create_housepvb(bus70,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h70=net.create_housepvb(bus71,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h71=net.create_housepvb(bus72,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h72=net.create_housepvb(bus73,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h73=net.create_housepvb(bus74,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h74=net.create_housepvb(bus75,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h75=net.create_housepvb(bus76,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h76=net.create_housepvb(bus77,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h77=net.create_housepvb(bus78,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h78=net.create_housepvb(bus79,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h79=net.create_housepvb(bus80,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h80=net.create_housepvb(bus81,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h81=net.create_housepvb(bus82,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h82=net.create_housepvb(bus83,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h83=net.create_housepvb(bus84,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h84=net.create_housepvb(bus85,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h85=net.create_housepvb(bus86,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h86=net.create_housepvb(bus87,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h87=net.create_housepvb(bus88,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h88=net.create_housepvb(bus89,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h89=net.create_housepvb(bus90,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h90=net.create_housepvb(bus91,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h91=net.create_housepvb(bus92,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h92=net.create_housepvb(bus93,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h93=net.create_housepvb(bus94,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h94=net.create_housepvb(bus95,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h95=net.create_housepvb(bus96,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h96=net.create_housepvb(bus97,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h97=net.create_housepvb(bus98,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h98=net.create_housepvb(bus99,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h99=net.create_housepvb(bus100,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h100=net.create_housepvb(bus101,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h101=net.create_housepvb(bus102,p_w=400,pv="pv_2kw",storage="battery_4kwh")
_,h102=net.create_housepvb(bus103,p_w=400,pv="pv_2kw",storage="battery_4kwh")
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
net.create_line(bus22,bus52,length_km=1,direct="down",theta=-90)
net.create_line(bus52,bus53,length_km=1)
net.create_line(bus53,bus54,length_km=1)
net.create_line(bus54,bus55,length_km=1)
net.create_line(bus55,bus56,length_km=1)
net.create_line(bus56,bus57,length_km=1)
net.create_line(bus57,bus58,length_km=1)
net.create_line(bus58,bus59,length_km=1)
net.create_line(bus59,bus60,length_km=1)
net.create_line(bus60,bus61,length_km=1)
net.create_line(bus22,bus62,length_km=1,direct="down",theta=-180)
net.create_line(bus62,bus63,length_km=1,theta=-180)
net.create_line(bus63,bus64,length_km=1,theta=-180)
net.create_line(bus64,bus65,length_km=1,theta=-180)
net.create_line(bus65,bus66,length_km=1,theta=-180)
net.create_line(bus66,bus67,length_km=1,theta=-180)
net.create_line(bus67,bus68,length_km=1,theta=-180)
net.create_line(bus68,bus69,length_km=1,theta=-180)
net.create_line(bus69,bus70,length_km=1,theta=-180)
net.create_line(bus70,bus71,length_km=1,theta=-180)
net.create_line(bus52,bus72,length_km=1,direct="down",theta=-90)
net.create_line(bus72,bus73,length_km=1)
net.create_line(bus73,bus74,length_km=1)
net.create_line(bus74,bus75,length_km=1)
net.create_line(bus75,bus76,length_km=1)
net.create_line(bus76,bus77,length_km=1)
net.create_line(bus77,bus78,length_km=1)
net.create_line(bus78,bus79,length_km=1)
net.create_line(bus79,bus80,length_km=1)
net.create_line(bus80,bus81,length_km=1)
net.create_line(bus52,bus82,length_km=1,direct="down",theta=-180)
net.create_line(bus82,bus83,length_km=1,theta=-180)
net.create_line(bus83,bus84,length_km=1,theta=-180)
net.create_line(bus84,bus85,length_km=1,theta=-180)
net.create_line(bus85,bus86,length_km=1,theta=-180)
net.create_line(bus86,bus87,length_km=1,theta=-180)
net.create_line(bus87,bus88,length_km=1,theta=-180)
net.create_line(bus88,bus89,length_km=1,theta=-180)
net.create_line(bus89,bus90,length_km=1,theta=-180)
net.create_line(bus90,bus91,length_km=1,theta=-180)
net.create_line(bus72,bus92,length_km=1,direct="down",theta=-180)
net.create_line(bus92,bus93,length_km=1,theta=-180)
net.create_line(bus93,bus94,length_km=1,theta=-180)
net.create_line(bus94,bus95,length_km=1,theta=-180)
net.create_line(bus95,bus96,length_km=1,theta=-180)
net.create_line(bus96,bus97,length_km=1,theta=-180)
net.create_line(bus97,bus98,length_km=1,theta=-180)
net.create_line(bus98,bus99,length_km=1,theta=-180)
net.create_line(bus99,bus100,length_km=1,theta=-180)
net.create_line(bus100,bus101,length_km=1,theta=-180)
#create_schem(net)
group1=[h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20]
group2=[h21,h22,h23,h24,h25,h26,h27,h28,h29,h30,h31,h32,h33,h34,h35,h36,h37,h38,h39,h40]
group3=[h41,h42,h43,h44,h45,h46,h47,h48,h49,h50,h51,h52,h53,h54,h55,h56,h57,h58,h59,h60]
group4=[h61,h62,h63,h64,h65,h66,h67,h68,h69,h70,h71,h72,h73,h74,h75,h76,h77,h78,h79,h80]
group5=[h81,h82,h83,h84,h85,h86,h87,h88,h89,h90,h91,h92,h93,h94,h95,h96,h97,h98,h99,h100,h101]

groups=[group1,group2,group3,group4,group5]
data=Data_intialize(net.net)
ls=Mathe_set(net.net,groups,Data_intialize)


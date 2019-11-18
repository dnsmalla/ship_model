
from schematic_draw.schemdraw import Drawing
from schematic_draw import elements as e
import numpy as np 
#d = schem.Drawing(unit=2.5)
direct=["down","up","left","right"]
def create_schem(net):
    data=net.plot_data
    #print(data)
    bus=net.bus_count
    circuit_data={}
    for i in range(bus):
        name='bus'+str(i)
        name=[]
        datas=data
        for d in datas[:]:
            if d[0]=='load':
                if d[1][1]==i:
                    name.append((d[0],d[1][1],d[2]))
                    data.remove(d)
            if d[0]=='storage':
                if d[1][1]==i:
                    name.append((d[0],d[1][1],d[2]))
                    data.remove(d)
            if d[0]=='ext_grid':
                if d[1][1]==i:
                    name.append((d[0],d[1][1]))
                    data.remove(d)
            if d[0]=='line':
                if d[1][1]==i:
                    name.append(('line',d[1][1],d[2][1],d[3][0],d[3][1]))
                    data.remove(d)
            if d[0]=='switch':
                if d[1][1]==i:
                    name.append(('switch',d[1][1],d[2][1],d[3][0],d[3][1]))
                    data.remove(d)
            if d[0]=='transformer':
                if d[1][1]==i:
                    name.append(('transformer',d[1][1],d[2][1]))
                    data.remove(d)
            if d[0]=='sgen':
                if d[1][1]==i:
                    name.append((d[0],d[1][1],d[2])) 
                    data.remove(d)
            if d[0]=='gen':
                if d[1][1]==i:
                    name.append((d[0],d[1][1],d[2])) 
                    data.remove(d)
        if len(name)>0: 
            circuit_data[str(i)]=name
    d = Drawing(unit=1,fontsize=9,lw=2)
    plot_data(d,circuit_data)


def plot_data(d,datas):
    start=[0,0]
    bus={}
    for key,val in datas.items():
        name='bus'+str(key)
        if not name in list(bus.keys()):
            value=Bus(name)
            bus[name]=value
        if bus[name]['start']is None :
            bus[name]['start']=[10,10]
        start=bus[name]['start']
        d.add(e.DOT,xy=start)
        for mem in val:
            if mem[0]=='ext_grid':
                now=d.add(e.GRID,d="up",xy=start,label='source')
                bus[name]['element'].append('ext_grid')
            if mem[0]=='line':
                direct=mem[3]
                theta=mem[4]
                bus[name]['element'].append('line')
                name='bus'+str(mem[2])
                if not name in list(bus.keys()):
                    value=Bus(mem[2])
                    bus[name]=value
                    now=d.add(e.LINE,d=direct,xy=start,l=2.75,theta=theta)
                    bus[name]['start']=now.end
                else:
                    end=bus[name]['start']
                    now=d.add(e.LINE,to=end,label='line')
                    bus[name]['start']=now.start
            if mem[0]=='sgen':
                direct=mem[2]
                now=d.add(e.PV,d=direct,xy=start,label='sgen')
                bus[name]['element'].append('sgen')

            if mem[0]=='gen':
                direct=mem[2]
                now=d.add(e.GEN,d=direct,xy=start,label='PV')
                bus[name]['element'].append('gen')

            if mem[0]=='storage':
                direct=mem[2]
                now=d.add(e.HOUSE,d=direct,xy=start,label='storage')
                bus[name]['element'].append('storage')

            if mem[0]=='load':
                if mem[2][1]==" ":
                    l_state=e.LOAD
                    lbl="Load"
                if mem[2][1]=="pv":
                    l_state=e.H_PV
                    lbl="PV"
                if mem[2][1]=="housepv":
                    l_state=e.HOUSEP
                    lbl="HP"
                if mem[2][1]=="house":
                    l_state=e.HOUSE
                    lbl="battery"
                if mem[2][1]=="housepvb":
                    l_state=e.HOUSEPB
                    lbl="HPB"
                direct=mem[2][0]
                now=d.add(l_state,d=direct,xy=start,label=lbl,lblloc='bot')
                bus[name]['element'].append('load')

            if mem[0]=='switch':
                if mem[3]:
                    state=e.SWITCH_OPEN
                else:
                    state=e.SWITCH_CLOSE
                direct=mem[4]
                now=d.add(state,d=direct,xy=start,l=1.75)
                bus[name]['element'].append('switch')
                name='bus'+str(mem[2])
                if not name in list(bus.keys()):
                    value=Bus(mem[2])
                    bus[name]=value
                    bus[name]['start']=now.end
                
            if mem[0]=='transformer':
                now=d.add(e.TRANS,d="down",xy=start,l=1.75,label='trans')
                bus[name]['element'].append('transformer')
                name='bus'+str(mem[2])
                if not name in list(bus.keys()):
                    value=Bus(mem[2])
                    bus[name]=value
                    bus[name]['start']=now.end
    d.draw()
    d.save("first_pic.pdf")

def Bus(name):
    name={}
    name['start']=None 
    name['end']=None 
    name['direstion']=None 
    name['element']=[]
    name['end']=False
    return name

# class Bus(object):
#     start=None 
#     end=None 
#     direction=None
#     element=[]
#     def __init__(self,id):
#         self.name='bus'+str(id)
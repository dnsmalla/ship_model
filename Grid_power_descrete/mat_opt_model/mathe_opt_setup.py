import sys
sys.path.append('./')
import pandas as pd
import numpy as np
import itertools
import time
import random
import copy
import math
from environments import Environment
import matplotlib.pyplot as plt

class Mathe_set():
    """to set up the learning models and network environment"""

    def __init__(self,net,group,reset):
        """agent set up for group learning
            agents name =agent+group_count
            agents[name]["name"]=agents in the group
            agent[policy]=policy that we use for learning
            call data_setup for getting data for learning
        """
        self.net=net
        self.reset=reset
        self.memory_size=20000
        self.total_groups=len([names for names in group])
        self.total_agents=len([name for names in group for name in names])
        self.all_names=[name for names in group for name in names]
        env=Environment()
        self.actions=["ON","OFF"]
        assert len(self.net.res_pv)==len(self.net.pv),"learning setup need res setup! import and setup data control "
        assert isinstance(group, list),"groput need to be list"
        self.agents={}
        self.reward={}
        for l in range(len(group)):
            name="agent"+str(l)
            g_name="group"+str(l)
            self.agents[name]={}
            self.agents[name]["group_name"]=g_name
            self.agents[name]["name"]=group[l]
            self.agents[name]["grid"]=0
            input_len=7
            action_len=2
        self.run(env)

    def run(self,env,train=True):
        """to run the all agent
            episodes
            time steps =24
        """
        start=time.time()
        env.train = True
        env.run_steps =1
        env.hour_max = 24
        for k in range(env.run_steps):
            env.step=k
            env.done=False
            for j in range(env.hour_max):
                print("this is time step",j)
                env.hour = j
                if j + 1 == env.hour_max:
                    env.done=True
                    env.next_hour = 0
                else:
                    env.next_hour = j + 1
                self.storage_can_discharge(env.hour,self.all_names)
                self.set_action(env)
                self.terminal_trig(env)
                if env.done:
                    break
            print("ended at ",j)
        self.plot_data(sow=True)

    def plot_data(self,sow=False):
        for i in range(len(self.agents)):
            agent="agent"+str(i) 
            names=self.agents[agent]["name"]
            soc_datas=self.net.res_storage_N_SOC.loc[names][:]
            load_data=(sum(list(self.net.res_load_data.loc[names][:].values)))
            pv_data=(sum(list(self.net.res_pv_production.loc[names][:].values)))
            grid_available=self.net.res_ext_grid.loc['Grid'][:]/self.total_groups
            grid_total_buy=sum(list(self.net.res_ext_grid_buy.loc[names][:].values))
            grid_total_sell=sum(list(self.net.res_ext_grid_2ld.loc[names][:].values)) + sum(list(self.net.res_ext_grid_2st.loc[names][:].values))
            if sow:
                plt.plot(soc_datas.T,label='agent', linewidth=3)
                plt.legend()
                plt.title(agent)
                plt.show()
                plt.plot(load_data,label='total_load_data', linewidth=3)
                plt.plot(pv_data)
                plt.plot(grid_available.T,label='grid_available', linewidth=3)
                plt.plot(grid_total_sell,label='totla sold by grid', linewidth=3)
                plt.plot(grid_total_buy,label='grid total buy', linewidth=3)
                plt.legend()
                plt.title(agent)
                plt.show()
            else:
                plt.plot(datas.T)
                plt.title(agent)
                plt.savefig("plot_result/"+agent + '.png')
        grid_total_buy=sum(list(self.net.res_ext_grid_buy.loc[self.all_names][:].values))
        grid_total_sell=sum(list(self.net.res_ext_grid_2ld.loc[self.all_names][:].values)) + sum(list(self.net.res_ext_grid_2st.loc[self.all_names][:].values))        
        all_load_data=(sum(list(self.net.res_load_data.loc[self.all_names][:].values)))
        all_pv_data=(sum(list(self.net.res_pv_production.loc[self.all_names][:].values)))
        all_grid_available=self.net.res_ext_grid.loc['Grid'][:]
        plt.plot(all_load_data,'m--o', linewidth=3,label="total load")
        plt.plot(all_pv_data,'c--o', linewidth=3,label="pv production")
        plt.plot(grid_total_buy,'g--o', linewidth=3,label="supply to Grid")
        plt.plot(grid_total_sell,'k--o', linewidth=3,label="total used form Grid")
        plt.plot(all_grid_available.T,'b--o', linewidth=3,label="total grid production")
        plt.title("total pv demand and power used form Grid")
        plt.legend()
        plt.show()


    def get_data_copy(self,leng,data):
        """to return the data as per length """
        return [data for i in range(leng)]

    def terminal_trig(self,env):
        hour=env.hour
        Hour="Hour-"+str(hour)
        usable_grid=self.net.res_ext_grid.loc['Grid'][hour]
        pv_available=sum(self.net.res_ext_grid_buy.loc[self.all_names][Hour])
        used_grid=self.grid_sell_all_call(hour,self.all_names)
        if used_grid > usable_grid+pv_available:
            env.done=True
    
    def set_action(self,env):
        hour=env.hour
        Hour="Hour-"+str(hour)
        use_data=self.net.res_storage_N_SOC.loc[self.all_names][Hour]
        can_discharge=use_data.sort_values(ascending=False)
        d_data=[]
        bc_data=[]
        bd_data=[]
        pv_data=[]
        for i in range(len(can_discharge)):
            name=can_discharge.keys()[i]
            d_data.append(self.net.res_load_data.at[name,hour])
            bc_data.append(self.net.res_can_charge.at[name,'p_w'])
            bd_data.append(self.net.res_can_discharge.at[name,'p_w'])
            pv_data.append(self.net.res_pv_production.at[name,hour])
        actions=self.get_action(bc_data,bd_data,d_data,pv_data,env)
        for i in range(len(can_discharge)):
            name=can_discharge.keys()[i]
            action=int(actions[i])
            self.implement_action(name,env,action)

    def get_action(self,charge_data,discharge_data,demand,pv_data,env):
        hour=env.hour
        grid=self.net.res_ext_grid.loc['Grid'][hour]
        pv_sum=sum(pv_data)
        if sum(charge_data)+sum(demand)<grid and pv_sum==0:
            print("in all_charging")
            return np.ones(len(charge_data))

        elif pv_sum>sum(demand)+500:
            print("all pv use")
            action=np.zeros(len(demand))
            print(action)
            return action

        elif grid+sum(discharge_data)<sum(demand):
            return "discharging and gird total does not fulfill demand"

        elif sum(demand)<grid:
            action=np.ones(len(demand))
            print("in charging")
            avg_charge=np.mean(charge_data)
            print("average charge",avg_charge)
            over_grid=grid-sum(demand)
            print("over_grid",over_grid)
            n=math.ceil(over_grid/avg_charge)
            for i in range(3,-1,-1):
                print(i)
                if (sum(charge_data[:n+i])+sum(demand)+sum(pv_data[:n+i]))<grid:
                    break  
            action[:n+i]=0
            print(action)
            return action

        elif sum(demand)>grid:
            action=np.ones(len(demand))
            print("in discharging")
            avg_demand=np.mean(demand)
            unfulfill_demand=sum(demand)-grid
            n=math.floor(unfulfill_demand/avg_demand)
            steps=len(demand)-n
            for j in range(steps):
                if sum(demand)<(grid+sum(demand[:n+j])+sum(pv_data[:n+j])):
                    break      
            action[:n+j]=0
            print(action)
            return action
        else:
            return "problem in solving action from get action"
        

    def implement_action(self,agent,env,action):
        """implement action
            for calculting next state data
        """
        hour=env.hour
        Hour="Hour-"+str(hour)
        load=self.net.res_load_data.at[agent,hour]
        pv  =self.net.res_pv_production.at[agent,hour]
        soc=self.net.res_storage_N_SOC.at[agent,Hour]
        storage_max=self.net.storage["max_p_w"][0]
        storage_min=self.net.storage["minimum_p_w"][0]
        if soc>1:
            grid_buy_from,grid_sell,pv_2sell,pv_2st,pv_2ld,st_2ld,st_4grid,st_4pv,load_4grid= \
                self.balance(storage_max,storage_min,soc,pv,load,action,hour)
            self.set_res_pv_2st(env,pv_2st,agent)
            self.set_res_pv_2sell(env,pv_2sell,agent)
            self.set_res_pv_2ld(env,pv_2ld,agent)
            self.set_res_ext_grid_2ld(env,load_4grid,agent)
            self.set_res_ext_grid_2st(env,st_4grid,agent)
            self.set_res_storage_2ld(env,st_2ld,agent)
            self.set_storage(env,agent)


    def pv_data_set(self,hour,name):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step group members pv data
        """
        return self.net.res_pv_production.loc[name][hour]

    def load_data_set(self,hour,name):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step group members load data
        """
        return self.net.res_load_data.loc[name][hour]

    def grid_sell_call(self,name,hour):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step all pv data
        """
        Hour = "Hour-"+str(hour)
        load_sell=self.net.res_ext_grid_2ld.loc[name][Hour]
        st_sell=self.net.res_ext_grid_2st.loc[name][Hour]
        return (np.sum(load_sell)+np.sum(st_sell))

    def grid_sell_all_call(self,hour,names):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step all pv data
        """
        Hour = "Hour-"+str(hour)
        load_sell=self.net.res_ext_grid_2ld.loc[names][Hour]
        st_sell=self.net.res_ext_grid_2st.loc[names][Hour]
        return (np.sum(load_sell)+np.sum(st_sell))

    def storage_can_discharge(self,hour,name):
        Hour="Hour-"+str(hour)
        SOC=self.net.res_storage_N_SOC.loc[name][Hour]
        can_discharge=self.net.res_storage_N_SOC.loc[name][Hour] - self.net.res_mini_charge.loc[name]["p_w"]
        can_discharge=(can_discharge[:]>0).astype(int)*can_discharge # select the discharabel agent
        can_discharge=(can_discharge[:]>2000).astype(int)*2000 +(can_discharge[:]<2000).astype(int)*can_discharge
        can_charge=self.net.res_max_charge.loc[name]["p_w"]-self.net.res_storage_N_SOC.loc[name][Hour]
        can_charge=(can_charge[:]>0).astype(int)*can_charge # select the discharabel agent
        can_charge=(can_charge[:]>2000).astype(int)*2000 +(can_charge[:]<2000).astype(int)*can_charge
        self.net.res_can_discharge["p_w"]=can_discharge
        self.net.res_can_charge["p_w"]=can_charge
        return can_discharge, can_charge

    def storage_data_set(self,hour,name):
        """fro group data return
        each time step [hour]
        but in net.res_storage_SOC there are "Hour-1" so convert in specific type
        group member [name]
        return specific time step group members storage data
        """
        Hour="Hour-"+str(hour)
        return self.net.res_storage_N_SOC.loc[name][Hour]

    def pv_data_call(self,hour,name):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step all pv data
        """
        return self.net.res_pv_production[hour][:]

    def storage_data_call(self,hour,name):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step group members storage data
        """
        Hour="Hour-"+str(hour)
        return self.net.res_storage_N_SOC[Hour][:]

    def load_data_call(self,hour):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step group members load data
        """
        return self.net.res_load_data[hour][:]

    def set_res_pv_2st(self,env, data,name):
        """to set the result to pv_res which set data to pv to storage and res_storge charge """
        Hour="Hour-"+str(env.hour)
        n_Hour="Hour-"+str(env.next_hour)
        self.net.res_pv_2st.loc[name][Hour]=data

    def set_res_pv_2ld(self,env, data,name):
        """to set the result to pv_res which set data to pv to load and load charge """
        Hour="Hour-"+str(env.hour)
        self.net.res_pv_2ld.at[name,Hour]=data
        self.net.res_load_4pv.at[name,Hour]=data
        self.net.res_load.loc[name]["pv_p_w"]=data

    def set_res_pv_2sell(self,env, data,name):
        """to set the result to pv_res which set data to pv to load and load charge """
        Hour="Hour-"+str(env.hour)
        self.net.res_pv_2sell.at[name,Hour]=data
        self.net.res_ext_grid_buy.at[name,Hour]=data

    def set_res_ext_grid_2ld(self,env, data,name):
        """to set the result to pv_res which set data to pv to load and load charge """
        Hour="Hour-"+str(env.hour)
        self.net.res_load.loc[name]['grid_p_w']=data
        self.net.res_load_4grid.at[name,Hour]=data
        self.net.res_ext_grid_2ld.at[name,Hour]=data

    def set_res_ext_grid_2st(self, env, data,name):
        """to set the result to pv_res which set data to pv to load and load charge """
        Hour="Hour-"+str(env.hour)
        n_Hour="Hour-"+str(env.next_hour)
        self.net.res_storage.loc[name]['grid_charge']=data
        self.net.res_ext_grid_2st.at[name,Hour]=data


    def set_res_storage_2ld(self, env, data,name):
        """to set the result to pv_res which set data to pv to load and load charge """
        Hour="Hour-"+str(env.hour)
        n_Hour="Hour-"+str(env.next_hour)
        self.net.res_storage_discharge.at[name,Hour]=data
        self.net.res_load_4st.at[name,Hour]=data
        self.net.res_load.loc[name]['st_p_w']=data

    def set_storage(self,env,name):
        """to set the soc value to the storage new state"""
        Hour="Hour-"+str(env.hour)
        if env.next_hour==0:
            n_Hour="Hour-"+str(env.hour+1)
        else:
            n_Hour="Hour-"+str(env.next_hour)
        self.net.res_storage_charge.at[name,Hour]=0.0
        self.net.res_storage_charge.at[name,Hour]=self.net.res_pv_2st.at[name,Hour]+self.net.res_ext_grid_2st.at[name,Hour]
        self.net.res_storage_N_SOC.at[name,n_Hour]=0.0
        self.net.res_storage_N_SOC.at[name,n_Hour]=self.net.res_storage_N_SOC.at[name,Hour]+self.net.res_storage_charge.at[name,Hour]
        self.net.res_storage_N_SOC.at[name,n_Hour]=self.net.res_storage_N_SOC.loc[name][n_Hour]-self.net.res_storage_discharge.at[name,Hour]

    def balance(self,storage_max,storage_min,soc,pv,load,action,hour,dt=1):
        """get data and return data  """
        
        action=self.actions[action]
        # print("action",action)
        # print("storage_max,storage_min,soc,pv,load,action,hour",storage_max,storage_min,soc,pv,load,action,hour)
        grid_buy_from =0
        grid_sell=0
        pv_2sell =0
        pv_2st   =0
        pv_2ld   =0
        st_2ld   =0
        st_4grid =0
        st_4pv   =0
        load_4grid=0

        
        if (storage_max - soc) > 2000*dt :
            storage_need=2000*dt
        elif storage_max <= soc:
            storage_need=0
        else:
            storage_need=(storage_max - soc)*dt

        if soc > storage_min and storage_max>soc:
            storage_dischargeable=(soc-storage_min)*dt
        else:
            storage_dischargeable=0

        if pv>load:
            pv_over=pv-load
            pv_nfill=0
        else:
            pv_over=0
            pv_nfill=load-pv

        if action == 'ON':
            if pv_over > storage_need:
                #print("1")
                grid_buy_from=pv_over
                grid_sell=0
                pv_2sell =pv_over
                pv_2st   =storage_need*dt
                pv_2ld   =load
                st_2ld   =0
                st_4grid =0
                st_4pv   =storage_need*dt
                load_4grid=0

            elif (pv_over>0) and (pv_over < storage_need):
                #print("2")
                grid_buy_from=0
                grid_sell=0
                pv_2sell =0
                pv_2st   =pv_over*dt
                pv_2ld   =load
                st_2ld   =0
                st_4grid =0
                st_4pv   =pv_over*dt
                load_4grid=0

            elif pv_nfill>0 and storage_dischargeable>pv_nfill:
                #print("3")
                grid_buy_from=0
                grid_sell=0
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =pv_nfill*dt
                st_4grid =0
                st_4pv   =0
                load_4grid=0

            elif storage_dischargeable>0 and storage_dischargeable<pv_nfill:
                #print("3")
                grid_buy_from=0
                grid_sell=load-storage_dischargeable*dt
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =storage_dischargeable
                st_4grid =0
                st_4pv   =0
                load_4grid=load-storage_dischargeable*dt

            elif storage_need==0:
                #print("4")
                grid_buy_from=0
                grid_sell=0
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =load
                st_4grid =0
                st_4pv   =0
                load_4grid=0
            else:
                #print("5")
                grid_buy_from=0
                grid_sell=load-storage_need
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =0
                st_4grid =storage_need
                st_4pv   =0
                load_4grid=load


        elif action == 'OFF':
            if pv >load:
                grid_buy_from=pv-load
                grid_sell= 0
                pv_2sell =pv-load
                pv_2st   =0
                pv_2ld   =load
                st_2ld   =0
                st_4grid =0
                st_4pv   =0
                load_4grid=0

            else:
                grid_buy_from=0
                grid_sell= load-pv
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =0
                st_4grid =0
                st_4pv   =0
                load_4grid=load-pv
        # print("grid_buy_from",grid_buy_from)
        # print("grid_sell",grid_sell)
        # print("pv_2sell",pv_2sell)
        # print("pv_2st",pv_2st)
        # print("pv_2ld",pv_2ld)
        # print("st_2ld",st_2ld)
        # print("st_4grid",st_4grid)
        # print("st_4pv",st_4pv)
        # print("load_4grid",load_4grid)
        return  grid_buy_from,grid_sell,pv_2sell,pv_2st,pv_2ld,st_2ld,st_4grid,st_4pv,load_4grid
    
    def balance_new(self,storage_max,storage_min,soc,pv,load,action,hour,dt=1):
        """get data and return data  """
        
        action=self.actions[action]
        # print("action",action)
        # print("storage_max,storage_min,soc,pv,load,action,hour",storage_max,storage_min,soc,pv,load,action,hour)
        grid_buy_from =0
        grid_sell=0
        pv_2sell =0
        pv_2st   =0
        pv_2ld   =0
        st_2ld   =0
        st_4grid =0
        st_4pv   =0
        load_4grid=0

        
        if (storage_max - soc) > 2000*dt :
            storage_need=2000*dt
        elif storage_max <= soc:
            storage_need=0
        else:
            storage_need=(storage_max - soc)*dt

        if soc > storage_min and storage_max>soc:
            storage_dischargeable=(soc-storage_min)*dt
        else:
            storage_dischargeable=0

        if pv>load:
            pv_over=pv-load
            pv_nfill=0
        else:
            pv_over=0
            pv_nfill=load-pv

        if action == 'ON':
            if hour>=22:
                grid_buy_from=0
                grid_sell=storage_need*dt+pv_nfill
                pv_2sell =0
                pv_2st   =pv_over
                pv_2ld   =0
                st_2ld   =0
                st_4grid =storage_need*dt
                st_4pv   =pv_over
                load_4grid=0

            elif pv_over > storage_need:
                grid_buy_from=pv_over
                grid_sell=0
                pv_2sell =pv_over
                pv_2st   =storage_need*dt
                pv_2ld   =load
                st_2ld   =0
                st_4grid =0
                st_4pv   =storage_need*dt
                load_4grid=0

            elif (pv_over>0) and (pv_over < storage_need):
                grid_buy_from=0
                grid_sell=0
                pv_2sell =0
                pv_2st   =pv_over*dt
                pv_2ld   =load
                st_2ld   =0
                st_4grid =0
                st_4pv   =pv_over*dt
                load_4grid=0

            elif pv_nfill>0 and storage_dischargeable>pv_nfill:
                grid_buy_from=0
                grid_sell=0
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =pv_nfill*dt
                st_4grid =0
                st_4pv   =0
                load_4grid=0

            elif storage_dischargeable>0 and storage_dischargeable<pv_nfill:
                grid_buy_from=0
                grid_sell=load-storage_dischargeable*dt
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =storage_dischargeable
                st_4grid =0
                st_4pv   =0
                load_4grid=load-storage_dischargeable*dt

            else:
                grid_buy_from=0
                grid_sell=load-storage_need
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =0
                st_4grid =storage_need
                st_4pv   =0
                load_4grid=load

        elif action == 'OFF':
            if pv >load:
                grid_buy_from=pv-load
                grid_sell= 0
                pv_2sell =pv-load
                pv_2st   =0
                pv_2ld   =load
                st_2ld   =0
                st_4grid =0
                st_4pv   =0
                load_4grid=0

            else:
                grid_buy_from=0
                grid_sell= load-pv
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =0
                st_4grid =0
                st_4pv   =0
                load_4grid=load-pv
        # print("grid_buy_from",grid_buy_from)
        # print("grid_sell",grid_sell)
        # print("pv_2sell",pv_2sell)
        # print("pv_2st",pv_2st)
        # print("pv_2ld",pv_2ld)
        # print("st_2ld",st_2ld)
        # print("st_4grid",st_4grid)
        # print("st_4pv",st_4pv)
        # print("load_4grid",load_4grid)
        return  grid_buy_from,grid_sell,pv_2sell,pv_2st,pv_2ld,st_2ld,st_4grid,st_4pv,load_4grid

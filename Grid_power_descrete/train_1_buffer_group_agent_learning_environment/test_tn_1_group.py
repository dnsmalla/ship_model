import sys
sys.path.append('./')
import pandas as pd
import numpy as np
import itertools
import time
import random
import copy
from environments import Environment
from DQN.q_single_target import Policy
from buffer import Memory
import matplotlib.pyplot as plt

class Test_group():
    """to set up the learning models and network environment"""

    def __init__(self,net,group,reset):
        """agent set up for group learning
            agents name =agent+group_count
            agents[name]["name"]=agents in the group
            agent[policy]=policy that we use for learning
            call data_setup for getting data for learning
        """
        self.net=net
        reset(self.net)
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
            self.agents[name]["memory"]=Memory(self.memory_size,input_len*2+9)
            for gm in range(len(group[l])):
                policy=group[l][gm]+"Policy"
                self.agents[name][policy]=Policy(input_len,action_len,group[l][gm],test=False)
        for l in range(len(group)):
            name="agent"+str(l)
            g_name="group"+str(l)
            for gm in range(len(group[l])):
                policy=group[l][gm]+"Policy"
                self.agents[name][policy].test_model()
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
            for j in range(24):
                env.hour = j
                if j + 1 == env.hour_max:
                    env.done=True
                    env.next_hour = 0
                else:
                    env.next_hour = j + 1

                for i in range(len(self.agents)):
                    agent="agent"+str(i)
                    self.get_action(agent,env)
                self.terminal_trig(env)
                if env.done:
                    print("group_episodes",k,"terminated at",j)
                    break
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
                plt.plot(soc_datas.T,label='agent')
                plt.legend()
                plt.title(agent)
                plt.show()
                plt.plot(load_data,label='total_load_data')
                plt.plot(pv_data)
                plt.plot(grid_available.T,label='grid_available')
                plt.plot(grid_total_sell,label='totla sold by grid')
                plt.plot(grid_total_buy,label='grid total buy')
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
        plt.plot(all_load_data)
        plt.plot(all_pv_data)
        plt.plot(grid_total_buy)
        plt.plot(grid_total_sell)
        plt.plot(all_grid_available.T)
        plt.title("total pv demand and power used form Grid")
        plt.show()

    def save_dict_to_file(self,dic):
        f = open('dictddpg.txt','w')
        f.write(str(dic))
        f.close()

    def load_dict_from_file(self):
        f = open('dictddpg.txt','r')
        data=f.read()
        f.close()
        return eval(data)

    def set_input(self,agent,env):
        """for hourly data set
         pv , storage , load
        load all agents members data from pv , storage , load
        return list of all data
        """
        data=[]
        hour=env.hour
        assert agent in list(self.agents.keys())," the agent is not containt in the agents list"
        names=self.agents[agent]["name"]
        data.append(list(self.pv_data_set(hour,names)/1000))
        data.append(list(self.load_data_set(hour,names)/1000))
        data.append(list(self.storage_data_set(hour,names)/1000))
        #avg_grid and time
        data.append(list(self.get_data_copy(len(names),self.net.res_ext_grid.loc['Grid'][hour]/self.total_agents/1000)))
        data.append(list(self.get_data_copy(len(names),(env.hour+1/24))))
        data=list(itertools.chain(*data))
        data=np.reshape(data,[5,-1])
        data[np.isnan(data)] = 0
        return data


    def set_next_put(self,agent,env):
        """for hourly data set
         pv , storage , load
        load all agents members data from pv , storage , load
        return list of all data
        """
        data=[]
        hour=env.next_hour
        assert agent in list(self.agents.keys())," the agent is not containt in the agents list"
        names=self.agents[agent]["name"]
        data.append(list(self.pv_data_set(hour,names)/1000))
        data.append(list(self.load_data_set(hour,names)/1000))
        data.append(list(self.storage_data_set(hour,names)/1000))
        #avg_grid and time
        data.append(list(self.get_data_copy(len(names),self.net.res_ext_grid.loc['Grid'][hour]/self.total_agents/1000)))
        data.append(list(self.get_data_copy(len(names),(env.hour+1/24))))
        data=list(itertools.chain(*data))
        data=np.reshape(data,[5,-1])
        data[np.isnan(data)] = 0
        return data

    def get_action(self,agent,env):
        """implement the data to get the action
        group data to get group action
        """
        data=self.set_input(agent,env)
        names=self.agents[agent]["name"]
        grid_now=0
        memory=self.agents[agent]["memory"]
        for index, name in np.ndenumerate(names):
            use_data=[]
            now=index[0]
            policy=names[now]+"Policy"
            demand_add=data[1]
            grid_total=data[3,1]
            if index[0]==0:
                grid_now=grid_total
            else:
                t_grid_sell=self.grid_sell_call(names[now-1],env.hour)/1000
                grid_now=grid_now - t_grid_sell
            use_data.append(sum(demand_add[now:]))
            use_data.append(grid_now)
            use_idata=list(data[:,now])
            used_data=use_data+use_idata
            used_data=np.reshape(used_data,[1,7])
            state=copy.copy(used_data)
            action=self.agents[agent][policy].choose_action(state)
            self.implement_action(names[now],env,action)
            next_data=self.set_next_put(agent,env)
            use_indata=list(next_data[:,now])
            n_state=use_data+use_indata
            n_state=np.reshape(n_state,[1,7])
            reward=self.cal_ireward(agent,names[now],env)
            g_reward=self.cal_greward(env,names)
            self.agents[agent][policy].learn_act(used_data,reward,n_state,env.done,g_reward,memory)

    def get_data_copy(self,leng,data):
        """to return the data as per length """
        return [data for i in range(leng)]


    def cal_ireward(self,agents,agent,env):
        """to return the reward"""
        #for individual reward
        hour=env.hour
        usable_igrid=self.net.res_ext_grid.loc['Grid'][hour]/self.total_agents
        used_igrid=self.grid_sell_call(agent,hour)
        if used_igrid>usable_igrid:
            ireward=-(used_igrid)/(usable_igrid+1)
        else:
            ireward=0.1

        return ireward

    def cal_greward(self,env,name):
        """to return the reward"""
        #for individual reward
        hour=env.hour
        usable_grid=self.net.res_ext_grid.loc['Grid'][hour]/len(name)
        used_grid=self.grid_sell_all_call(hour,name)
        if used_grid > usable_grid:
            g_reward=-1
        else:
            g_reward=0.1
        return g_reward
    
    def terminal_trig(self,env):
        hour=env.hour
        usable_grid=self.net.res_ext_grid.loc['Grid'][hour]
        used_grid=self.grid_sell_all_call(hour,self.all_names)
        if used_grid > usable_grid:
            env.done=True

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
            grid_buy,grid_sell,pv_2sell,pv_2st,pv_2ld,st_2ld,st_4grid,st_4pv,load_4grid= \
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
        print("action",action)
        # print("storage_max,storage_min,soc,pv,load,action,hour",storage_max,storage_min,soc,pv,load,action,hour)
        grid_buy =0
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
                grid_buy=pv_over
                grid_sell=0
                pv_2sell =pv_over
                pv_2st   =storage_need*dt
                pv_2ld   =load
                st_2ld   =0
                st_4grid =0
                st_4pv   =storage_need*dt
                load_4grid=0

            elif (pv_over>0) and (pv_over < storage_need):
                grid_buy=0
                grid_sell=0
                pv_2sell =0
                pv_2st   =pv_over*dt
                pv_2ld   =load
                st_2ld   =0
                st_4grid =0
                st_4pv   =pv_over*dt
                load_4grid=0

            elif pv_nfill>0 and storage_dischargeable>pv_nfill:
                grid_buy=0
                grid_sell=0
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =pv_nfill*dt
                st_4grid =0
                st_4pv   =0
                load_4grid=0

            elif storage_dischargeable>0 and storage_dischargeable<pv_nfill:
                grid_buy=0
                grid_sell=load-storage_dischargeable*dt
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =storage_dischargeable
                st_4grid =0
                st_4pv   =0
                load_4grid=load-storage_dischargeable*dt

            else:
                grid_buy=0
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
                grid_buy=pv-load
                grid_sell= 0
                pv_2sell =pv-load
                pv_2st   =0
                pv_2ld   =load
                st_2ld   =0
                st_4grid =0
                st_4pv   =0
                load_4grid=0

            else:
                grid_buy=0
                grid_sell= load-pv
                pv_2sell =0
                pv_2st   =0
                pv_2ld   =pv
                st_2ld   =0
                st_4grid =0
                st_4pv   =0
                load_4grid=load-pv
        # print("grid_buy",grid_buy)
        # print("grid_sell",grid_sell)
        # print("pv_2sell",pv_2sell)
        # print("pv_2st",pv_2st)
        # print("pv_2ld",pv_2ld)
        # print("st_2ld",st_2ld)
        # print("st_4grid",st_4grid)
        # print("st_4pv",st_4pv)
        # print("load_4grid",load_4grid)
        return  grid_buy,grid_sell,pv_2sell,pv_2st,pv_2ld,st_2ld,st_4grid,st_4pv,load_4grid

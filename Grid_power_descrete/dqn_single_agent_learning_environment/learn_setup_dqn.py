import pandas as pd
import numpy as np
import itertools
import time
import random
from environments import Environment
from DQN.dqn import Policy

class Learn_set():
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
        self.total_agents=len([name for names in group for name in names])
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
            for gm in range(len(group[l])):
                policy=group[l][gm]+"Policy"
                self.agents[name][policy]=Policy(input_len,action_len,group[l][gm],test=False)
        self.run(env)

    def run(self,env,train=True):
        """to run the all agent
            episodes
            time steps =24
        """
        start=time.time()
        env.train = True
        env.run_steps =50000
        env.hour_max = 24
        for k in range(env.run_steps):

            for j in range(24):
                env.hour = j
                if j + 1 == env.hour_max:
                    env.done=True
                    env.next_hour = 0
                else:
                    env.done=False
                    env.next_hour = j + 1

                for i in range(len(self.agents)):
                    agent="agent"+str(i)
                    self.agents[agent]["grid"]=0
                    learn_agent=list(self.agents[agent]["name"])
                    learn_agent = random.sample(learn_agent, k=len(learn_agent))
                    for ags in learn_agent:
                        policy=self.agents[agent][ags+"Policy"]
                        input,action=self.get_action(agent,ags,env)
                        next_s,reward=self.get_renex(agent,ags,env)
                        g_reward=self.cal_greward(env,learn_agent)

                        policy.learn_act(input,reward,next_s,env.done,g_reward)
                        if k+1>env.run_steps-5 and j+1==24:
                            policy.save_model()
                        # if j+1==24:
                        #     print(" steps",k,"  agent ",ags," reward ",reward)
                if env.done:
                    print("dqn_episodes",k,"terminated at",j)
                    break
            self.reward[str(k)]=j
            self.reset(self.net)
            now=time.time()
            #print("time taken",now-start)
        self.save_dict_to_file(self.reward)

    def save_dict_to_file(self,dic):
        f = open('dictdqn.txt','w')
        f.write(str(dic))
        f.close()

    def load_dict_from_file(self):
        f = open('dictdqn.txt','r')
        data=f.read()
        f.close()
        return eval(data)

    def set_input(self,times,agent,env):
        """for hourly data set
         pv , storage , load
        load all agents members data from pv , storage , load
        return list of all data
        """
        data=[]
        hour=env.hour
        data.append(self.pv_data_set(hour,agent)/1000)
        data.append(self.load_data_set(hour,agent)/1000)
        data.append(self.storage_data_set(hour,agent)/1000)
        #avg_grid and time
        data.append(self.net.res_ext_grid.loc['Grid'][hour]/self.total_agents/1000)
        data.append((env.hour))
        if data[1]>1:
            data.append(1)
        else:
            data.append(0)
        if data[1]<data[2]:
            data.append(0)
        else:
            data.append(-1)
        data=np.reshape(data,[1,len(data)])
        data[np.isnan(data)] = 0
        return data


    #Todo
    def get_action(self,agents,agent,env):
        """implement the data to get the action
        group data to get group action
        """
        agent_len=len(self.agents[agents]["name"])
        data=self.set_input(agent_len,agent,env)
        policy=agent+"Policy"
        action=self.agents[agents][policy].choose_action(data)
        self.implement_action(agent,env,action)
        return data,action
    
    def get_l_action(self,agents,agent,env):
        """implement the data to get the action
        group data to get group action
        """
        agent_len=len(self.agents[agents]["name"])
        data=self.set_input(agent_len,agent,env)
        policy=agent+"Policy"
        action=self.agents[agents][policy].choose_action(data)
        self.implement_action(agent,env,action)
        return data,action

    def get_data_copy(self,leng,data):
        """to return the data as per length """
        return [data for i in range(leng)]


    def get_renex(self,agents,agent,env):
        """input as a agent which cotaint group name
            return reward and next state
        """
        agent_len=len(self.agents[agents]["name"])
        reward=self.cal_ireward(agent_len,agents,agent,env)
        data=self.cal_next_state(agent_len,agent,env)
        
        return data,reward

    def cal_next_state(self,times,agent,env):
        hour=env.next_hour
        data=[]
        data.append(self.pv_data_set(hour,agent)/1000)
        data.append(self.load_data_set(hour,agent)/1000)
        data.append(self.storage_data_set(hour,agent)/1000)
        #grid_avg and time
        data.append(self.net.res_ext_grid.loc['Grid'][hour]/self.total_agents/1000)
        data.append((env.next_hour))
        if data[1]>1:
            data.append(1)
        else:
            data.append(0)
        if data[1]<data[2]:
            data.append(0)
        else:
            data.append(-1)
        data=np.reshape(data,[1,len(data)])
        data[np.isnan(data)] = 0
        return data


    def cal_ireward(self,times,agents,agent,env):
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
            env.done=True
            g_reward=-1
        else:
            g_reward=0.1
        return g_reward

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
        return self.net.res_pv_production.at[name,hour]

    def load_data_set(self,hour,name):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step group members load data
        """
        return self.net.res_load_data.at[name,hour]

    def storage_data_set(self,hour,name):
        """fro group data return
        each time step [hour]
        but in net.res_storage_SOC there are "Hour-1" so convert in specific type
        group member [name]
        return specific time step group members storage data
        """
        Hour="Hour-"+str(hour)
        return self.net.res_storage_N_SOC.at[name,Hour]

    def pv_data_call(self,hour,name):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step all pv data
        """
        return self.net.res_pv_production[hour][:]

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

    def grid_sell_call(self,name,hour):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step all pv data
        """
        Hour = "Hour-"+str(hour)
        load_sell=self.net.res_ext_grid_2ld.loc[name][Hour]
        st_sell=self.net.res_ext_grid_2st.loc[name][Hour]
        return (load_sell+st_sell)

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
    #todo
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
    #todo

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
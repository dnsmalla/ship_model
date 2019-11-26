import pandas as pd
import numpy as np
import itertools
import time
from environments import Environment
from DDPG import DDPGAgents

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
        self.reward={}
        self.reset=reset
        self.total_agents=len([name for names in group for name in names])
        self.total_groups=len([names for names in group] )
        env=Environment()
        self.actions=["ON","OFF"]
        assert len(self.net.res_pv)==len(self.net.pv),"learning setup need res setup! import and setup data control "
        assert isinstance(group, list),"groput need to be list"
        self.agents={}
        for l in range(len(group)):
            name="agent"+str(l)
            g_name="group"+str(l)
            self.agents[name]={}
            self.agents[name]["group_name"]=g_name
            self.agents[name]["name"]=group[l]
            input_len=5*len(group[l])
            action_len=len(group[l])
            self.agents[name]["Policy"]=DDPGAgents.Policy(input_len,action_len,g_name)
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
                    input,action=self.get_action(agent,env)
                    next_s,reward=self.get_renex(agent,env)
                    g_reward=self.cal_greward(env)
                    self.agents[agent]["Policy"].learn_act(input[0],reward,next_s[0],env.done,g_reward)
                    if k+1>env.run_steps-10 and j+1==env.hour_max:
                        self.agents[agent]["Policy"].save_model()
                    # if j+1==24:
                    #     print(" steps",k,"  agent ",agent," reward ",reward)
                if env.done:
                    print("episodes",k,"terminated at",j)
                    break
            self.reward[str(k)]=j
            self.reset(self.net)
            now=time.time()
            #print("time taken",now-start)
        self.save_dict_to_file(self.reward)

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

        data.append(list(self.get_data_copy(len(names),self.net.res_ext_grid.loc['Grid'][hour]/self.total_groups/1000)))
        data.append(list(self.get_data_copy(len(names),(env.hour))))

        data=list(itertools.chain(*data))
        data=np.reshape(data,[1,len(data)])
        data[np.isnan(data)] = 0
        return data

    def get_action(self,agent,env):
        """implement the data to get the action
        group data to get group action
        """
        data=self.set_input(agent,env)
        action=self.agents[agent]["Policy"].choose_action(data,env.step)
        self.implement_action(agent,env,action)
        return data,action

    def get_data_copy(self,leng,data):
        """to return the data as per length """
        return [data for i in range(leng)]


    def get_renex(self,agent,env):
        """input as a agent which cotaint group name
            return reward and next state
        """
        ireward=self.cal_ireward(agent,env)
        next_state=self.cal_next_state(agent,env)
        return next_state,ireward

    def cal_next_state(self,agent,env):
        hour=env.next_hour
        data=[]
        assert agent in list(self.agents.keys())," the agent is not containt in the agents list"
        names=self.agents[agent]["name"]
        data.append(list(self.pv_data_set(hour,names)/1000))
        data.append(list(self.load_data_set(hour,names)/1000))
        data.append(list(self.storage_data_set(hour,names)/1000))
        #grid_avg and time
        data.append(list(self.get_data_copy(len(names),self.net.res_ext_grid.loc['Grid'][hour]/self.total_groups/1000)))
        data.append(list(self.get_data_copy(len(names),(env.next_hour))))
        data=list(itertools.chain(*data))
        data=np.reshape(data,[1,len(data)])
        data[np.isnan(data)] = 0
        return data


    def cal_ireward(self,agent,env):
        """to return the reward
            agent =group
        """
        hour=env.hour
        usable_igrid,used_igrid=0,0
        group_len=len(self.agents[agent]["name"])
        usable_igrid=(self.net.res_ext_grid.loc['Grid'][hour]/self.total_agents)*group_len
        used_igrid=self.grid_sell_call(self.agents[agent]["name"],hour)
        if used_igrid>usable_igrid:
            ireward=-(used_igrid)/(usable_igrid)
        else:
            ireward=0.1
        return ireward

    def cal_greward(self,env):
        """to return the reward
            agent =group
        """
        hour=env.hour
        usable_grid,used_grid=0,0
        usable_grid=self.net.res_ext_grid.loc['Grid'][hour]
        used_grid=self.grid_sell_all_call(hour)
        if used_grid>usable_grid:
            env.done=True
            greward=-(used_grid)/(usable_grid)
        else:
            greward=0
        return greward

    def implement_action(self,agent,env,actions):
        """implement action
            for calculting next state data
        """
        hour=env.hour
        names=self.agents[agent]["name"]
        assert len(names)==len(actions),"action length is not sufficient to implement all agents"
        Hour="Hour-"+str(hour)

        for j in range(len(names)):
            load=self.net.res_load_data.at[names[j],hour]
            pv  =self.net.res_pv_production.at[names[j],hour]
            soc=self.net.res_storage_N_SOC.at[names[j],Hour]
            storage_max=self.net.storage["max_p_w"][0]
            storage_min=self.net.storage["minimum_p_w"][0]
            if soc>1:
                grid_buy,grid_sell,pv_2sell,pv_2st,pv_2ld,st_2ld,st_4grid,st_4pv,load_4grid= \
                    self.balance(storage_max,storage_min,soc,pv,load,actions[j],hour)
                self.set_res_pv_2st(env,pv_2st,names[j])
                self.set_res_pv_2sell(env,pv_2sell,names[j])
                self.set_res_pv_2ld(env,pv_2ld,names[j])
                self.set_res_ext_grid_2ld(env,load_4grid,names[j])
                self.set_res_ext_grid_2st(env,st_4grid,names[j])
                self.set_res_storage_2ld(env,st_2ld,names[j])
                self.set_storage(env,names[j])
            else:
                raise Exception("soc is nan")
                pass

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

    def grid_sell_all_call(self,hour):
        """fro group data return
        each time step [hour]
        group member [name]
        return specific time step all pv data
        """
        Hour = "Hour-"+str(hour)
        load_sell=self.net.res_ext_grid_2ld[Hour][:]
        st_sell=self.net.res_ext_grid_2st[Hour][:]
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
        n_Hour="Hour-"+str(env.next_hour)
        self.net.res_storage_charge.at[name,Hour]=0.0
        self.net.res_storage_charge.at[name,Hour]=self.net.res_pv_2st.at[name,Hour]+self.net.res_ext_grid_2st.at[name,Hour]
        self.net.res_storage_N_SOC.at[name,n_Hour]=0.0
        self.net.res_storage_N_SOC.at[name,n_Hour]=self.net.res_storage_N_SOC.at[name,Hour]+self.net.res_storage_charge.at[name,Hour]
        self.net.res_storage_N_SOC.at[name,n_Hour]=self.net.res_storage_N_SOC.loc[name][n_Hour]-self.net.res_storage_discharge.at[name,Hour]

    def balance(self,storage_max,storage_min,soc,pv,load,action,hour):
        """get data and return data  """
        if action>0:
            action=1
        else:
            action=0
        action=self.actions[action]
        grid_buy =0
        grid_sell=0
        pv_2sell =0
        pv_2st   =0
        pv_2ld   =0
        st_2ld   =0
        st_4grid =0
        st_4pv   =0
        load_4grid=0
        if action == 'ON':
            if hour >= 46:
                balance = storage_max - soc
                if balance>1000:
                    grid_buy=0
                    grid_sell= load+ 1000
                    pv_2sell = 0
                    pv_2ld   = 0
                    pv_2st   =0
                    st_2ld   =0
                    st_4grid =1000
                    st_4pv   =0
                    load_4grid=load
                elif balance>0:
                    grid_buy =0
                    grid_sell=load+balance
                    pv_2sell =0
                    pv_2st   =0
                    pv_2ld   =0
                    st_2ld   =0
                    st_4grid =balance
                    st_4pv   =0
                    load_4grid=load
                else:
                    raise Exception("storage soc is greater than it max capacity")
            else:
                if pv > load:
                    balance = storage_max - soc
                    pv_balance=pv-load
                    if soc >=storage_min+load:
                        grid_buy=pv
                        grid_sell=0
                        pv_2sell =pv
                        pv_2st   =0
                        pv_2ld   =0
                        st_2ld   =load
                        st_4grid =0
                        st_4pv   =0
                        load_4grid=0

                    else:
                        grid_buy=pv_balance
                        grid_sell=0
                        pv_2sell =pv_balance
                        pv_2st   =0
                        pv_2ld   =load
                        st_2ld   =0
                        st_4grid =0
                        st_4pv   =0
                        load_4grid=0


                elif pv > 0:
                    if soc >= storage_min+load:
                        grid_buy=pv
                        grid_sell=0
                        pv_2sell =pv
                        pv_2st   =0
                        pv_2ld   =0
                        st_2ld   =load
                        st_4grid =0
                        st_4pv   =0
                        load_4grid=0

                    else:
                        grid_buy=0
                        grid_sell= load-pv +1000
                        pv_2sell =0
                        pv_2st   =0
                        pv_2ld   =pv
                        st_2ld   =0
                        st_4grid =1000
                        st_4pv   =0
                        load_4grid=load-pv
                else:
                    balance = storage_max - soc
                    if balance>1000:
                        grid_buy=0
                        grid_sell= load+1000
                        pv_2sell =0
                        pv_2st   =0
                        pv_2ld   =0
                        st_2ld   =0
                        st_4grid =1000
                        st_4pv   =0
                        load_4grid=load
                    elif balance>=0:
                        grid_buy=0
                        grid_sell= load+balance
                        pv_2sell =0
                        pv_2st   =0
                        pv_2ld   =0
                        st_2ld   =0
                        st_4grid =balance
                        st_4pv   =0
                        load_4grid=load
                    else:
                        print(storage_max,soc)
                        raise Exception("storage soc is greater than it max capacity")

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

        return  grid_buy,grid_sell,pv_2sell,pv_2st,pv_2ld,st_2ld,st_4grid,st_4pv,load_4grid

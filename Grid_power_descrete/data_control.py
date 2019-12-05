import os
import copy
import pandas as pd 
import numpy as np 
from excel_read_write import Load_gen,PV_gen,Storage_gen,Grid_gen
data_file='model_data.xlsx'

class Data_intialize():
    """data intialize for network use"""

    def __init__(self,net):
        """input file load data to the load
                      pv production to pv 
                      battery starting as well
        """
        self.net=net
        assert os.path.exists, "file does not exit"
        assert os.path.splitext(data_file)[1]=='.xlsx', "file need to be name.xlsx"
        self.file=data_file 
        self.reset_results(suffix=None)
        self.upload_data(random=True)

    def upload_data(self,random=True):
        """ to setup the data for group """
        if random:
            #pv data generator
            pv_need=list(self.net.pv["name"])
            pv=PV_gen()
            pv_data=pv._datas(pv_need)
            self.net.res_pv_production=pv_data.T
            #load data generator
            load_need=list(self.net.load["name"])
            load=Load_gen()
            load_data=load._datas(load_need)
            self.net.res_load_data=load_data.T
            #load storage start soc
            soc_need=list(self.net.storage["name"])
            soc=Storage_gen()
            soc_data=soc._datas(soc_need)
            self.net.res_storage_SOC["Hour-0"]=soc_data.T
            self.net.res_storage_N_SOC["Hour-0"][:]=self.net.res_storage_SOC["Hour-0"]
            # load grid data
            grid_need=list(self.net.ext_grid["name"])
            grid=Grid_gen()
            grid_data=grid._datas(grid_need)
            self.net.res_ext_grid=grid_data.T

        else:
            load_datas= pd.read_excel(self.file,'load')
            pv_datas= pd.read_excel(self.file,'pv')
            #storage_datas= pd.read_excel(self.file,'battery',index_col=0)
            storage_datas= pd.read_excel(self.file,'battery')
            grid_datas= pd.read_excel(self.file,'grid')
            assert len(pv_datas)==24,"data canot feed to pv production"
            self.net.res_pv_production=pv_datas.T
            assert len(load_datas)==24,"data canot feed to load production"
            self.net.res_load_data=load_datas.T
            #assert len(self.net.res_storage_SOC)==len(storage_datas.values.copy()[0]),"storage data does note match with storage len"
            self.net.res_storage_SOC["Hour-0"][:]=copy.deepcopy(storage_datas.values.T[2])
            self.net.res_ext_grid=grid_data.T

    def get_elements_to_empty(self):
        return ["ext_grid_buy", "ext_grid_2st","ext_grid_2ld","ext_grid_balance","pv_2st","pv_2ld","pv_2sell","storage_charge","storage_discharge",
                "load","storage","pv_production","load_data","storage_SOC","load_4grid","load_4pv","load_4st","storage_N_SOC"]

    def get_elements_to_init(self):
       return ["ext_grid","gen","pv","house","housepv","housepvb"]

    def get_result_tables(self,element, suffix=None):
        res_empty_element = "_empty_res_" + element
        res_element = "res_" + element
        if suffix is not None:
            res_element += suffix
        return res_element, res_empty_element

    def empty_res_element(self, element, suffix=None):
        res_element, res_empty_element = self.get_result_tables(element, suffix)
        self.net[res_element] = self.net[res_empty_element].copy()

    def init_element(self, element, suffix=None):
        res_element, res_empty_element = self.get_result_tables(element, suffix)
        index = self.net[element].index
        if len(index):
            # init empty dataframe
            res_columns = self.net[res_empty_element].columns
            self.net[res_element] = pd.DataFrame(np.nan, index=index, columns=res_columns, dtype='float')
        else:
            self.empty_res_element(element, suffix)

    def reset_results(self, suffix=None):
        elements_to_empty = self.get_elements_to_empty()
        for element in elements_to_empty:
            self.empty_res_element(element, suffix)
        elements_to_init = self.get_elements_to_init()
        for element in elements_to_init:
            self.init_element(element, suffix)
        self.set_pv_res()
        self.set_storage_res()
        self.set_ext_grid_res()
        self.set_load_data_res()

    def set_pv_res(self):
        self.net["res_pv"]["name"]=self.net["load"]["name"]
        self.net["res_pv"]=self.net["res_pv"].set_index('name')
        self.net["res_pv_production"]["name"]=self.net["load"]["name"]
        self.net["res_pv_production"]=self.net["res_pv_production"].set_index('name')
        self.net["res_pv_2sell"]["name"]=self.net["load"]["name"]
        self.net["res_pv_2sell"]=self.net["res_pv_2sell"].set_index('name')
        self.net["res_pv_2st"]["name"]=self.net["load"]["name"]
        self.net["res_pv_2st"]=self.net["res_pv_2st"].set_index('name')
        self.net["res_pv_2ld"]["name"]=self.net["load"]["name"]
        self.net["res_pv_2ld"]=self.net["res_pv_2ld"].set_index('name')

    def set_storage_res(self):
        self.net["res_storage"]["name"]=self.net["load"]["name"]
        self.net["res_storage"]=self.net["res_storage"].set_index('name')
        self.net["res_storage_charge"]["name"]=self.net["load"]["name"]
        self.net["res_storage_charge"]=self.net["res_storage_charge"].set_index('name')
        self.net["res_storage_discharge"]["name"]=self.net["load"]["name"]
        self.net["res_storage_discharge"]=self.net["res_storage_discharge"].set_index('name')
        self.net["res_storage_SOC"]["name"]=self.net["load"]["name"]
        self.net["res_storage_SOC"]=self.net["res_storage_SOC"].set_index('name')
        self.net["res_storage_N_SOC"]["name"]=self.net["load"]["name"]
        self.net["res_storage_N_SOC"]=self.net["res_storage_N_SOC"].set_index('name')

    def set_ext_grid_res(self):
        self.net["res_ext_grid"]["name"]=self.net["ext_grid"]["name"]
        self.net["res_ext_grid"]=self.net["res_ext_grid"].set_index('name')
        self.net["res_ext_grid_buy"]["name"]=self.net["pv"]["name"]
        self.net["res_ext_grid_buy"]=self.net["res_ext_grid_buy"].set_index('name')
        self.net["res_ext_grid_2st"]["name"]=self.net["load"]["name"]
        self.net["res_ext_grid_2st"]=self.net["res_ext_grid_2st"].set_index('name')
        self.net["res_ext_grid_2ld"]["name"]=self.net["load"]["name"]
        self.net["res_ext_grid_2ld"]=self.net["res_ext_grid_2ld"].set_index('name')
        self.net["res_ext_grid_balance"]["name"]=self.net["ext_grid"]["name"]
        self.net["res_ext_grid_balance"]=self.net["res_ext_grid_balance"].set_index('name')

    def set_load_data_res(self):
        self.net["res_load"]["name"]=self.net["load"]["name"]
        self.net["res_load"]=self.net["res_load"].set_index('name')
        self.net["res_load_data"]["name"]=self.net["load"]["name"]
        self.net["res_load_data"]=self.net["res_load_data"].set_index('name')
        self.net["res_load_4grid"]["name"]=self.net["load"]["name"]
        self.net["res_load_4grid"]=self.net["res_load_4grid"].set_index('name')
        self.net["res_load_4pv"]["name"]=self.net["load"]["name"]
        self.net["res_load_4pv"]=self.net["res_load_4pv"].set_index('name')
        self.net["res_load_4st"]["name"]=self.net["load"]["name"]
        self.net["res_load_4st"]=self.net["res_load_4st"].set_index('name')
        


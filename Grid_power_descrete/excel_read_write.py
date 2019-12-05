import numpy as np
import pandas as pd

class Excel(object):
    def __init__(self,directory="data.xlsx",time_step=24,total_agent=None):
        self.directory=directory
        self.data={}
        self.time_step=time_step
        self.total_agents=total_agent

        if self.exist():
            for s in self.sheets():
                self.data[s]=self.get_sheet(s)
                column_data=self.data[s].columns
                if self.total_agents is None:
                    self.total_agents=column_data[:]

    def __getitem__(self,k):
        return self.data[k]

    def __setitem__(self,k,v):
        self.data[k]=v

    def get_sheet(self,name="",fail_accept=True):
        if fail_accept and name not in self.sheets():
            return None
        else:
            return pd.read_excel(self.directory,sheet_name=name)

    def sheets(self):
        """ read the sheets name"""
        return pd.ExcelFile(self.directory).sheet_names

    def read_excel(self,f):
        """ read the file"""
        return pd.read_excel(f)

    def read_sheet(self,f):
        """ give sheet name " "
        """
        assert f in self.sheets(),"named sheet is not in excel"
        return pd.read_excel(self.directory,sheet_name=f)

    def exist(self):
        """ to find out if the directory exist or not"""
        try:
            _ = open(self.directory)
            return True
        except FileNotFoundError:
            return False

    def save(self):
        """ to save the data items to excel file"""
        with pd.ExcelWriter(self.directory,engine='xlsxwriter') as writer:
            for k,v in self.data.items():
                v.to_excel(writer, sheet_name=k)



class Load_gen(object):

    def __init__(self,low_w=500,high_w=1700,dt_time=24):
        """ low_w is lower point of load power
            high_w is maximum point of load power
            dt_time is discretization of 24hour in to that form
        """
        self.low_w=low_w
        self.high_w=high_w
        self.dt_time=dt_time

    def _data(self,_head):
        """create data for load"""
        data={}
        load=np.random.randint(self.low_w,self.high_w,size=self.dt_time)
        data[_head]=load
        return pd.DataFrame(data,columns= [_head])

    def _datas(self,cols):
        load_cols=[]
        for col in range(len(cols)):
            if len(load_cols)==0:
                load_data=self._data(cols[col])
                load_cols.append(load_data.columns[0])
            else:
                load_datas=self._data(cols[col])
                if load_datas.columns[0] not in load_cols:
                    load_cols.append(load_datas.columns[0])
                    load_data=load_data.join(load_datas)

        return load_data


class PV_gen(object):

    def __init__(self,start_t=5,end_t=18,dt_time=24,max_pv=3000,min_pv=1000):
        """ low_w is min_pv is  minimum  pv production
            max_pv is maximum point of pv production power
            dt_time is discretization of 24hour in to that form
            start_t is pv production starting point
            end_t is pv production end time
        """
        self.start_t=start_t
        self.end_w=end_t
        self.dt_time=dt_time
        self.max_pv=max_pv
        self.min_pv=min_pv
        self.start_data=0
        self._data_init()

    def _data_init(self):
        """ bin is for normal distribution
            low_w is min_pv is  minimum  pv production
            max_pv is maximum point of pv production power
            dt_time is discretization of 24hour in to that form
            start_t is pv production starting point
            end_t is pv production end time
        """
        bins=[-0.30051769, -0.27382177, -0.24712585, -0.22042993, -0.19373401, -0.16703809, -0.14034217,
              -0.11364625, -0.08695033, -0.06025441, -0.0335585,  -0.00686258, 0.01983334,  0.04652926,
              0.07322518,  0.0999211,   0.12661702,  0.15331294, 0.18000886,  0.20670478 , 0.23340069,
              0.26009661,  0.28679253,  0.31348845]
        start_zero=list(np.zeros(self.start_t))
        mid_one=list(np.ones(self.end_w-self.start_t))
        last_zero=list(np.zeros(self.dt_time-self.end_w))
        self.start_data=start_zero+mid_one+last_zero
        mu, sigma = 0, 0.1
        a=[]
        for i in range(24):
            a.append(1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins[i] - mu)**2 / (2 * sigma**2) ))
        a=np.array(a)
        self.start_data= np.array(self.start_data)*a/4

    def _data(self,_head):
        """create data for load"""
        data={}
        pv=np.random.randint(self.min_pv,self.max_pv,size=self.dt_time)
        pv=pv*self.start_data
        data[_head]=pv
        return pd.DataFrame(data,columns= [_head])

    def _datas(self,cols):
        """ to create random  data for all element data that are in pv name"""
        load_cols=[]
        for col in range(len(cols)):
            if len(load_cols)==0:
                load_data=self._data(cols[col])
                load_cols.append(load_data.columns[0])
            else:
                load_datas=self._data(cols[col])
                if load_datas.columns[0] not in load_cols:
                    load_cols.append(load_datas.columns[0])
                    load_data=load_data.join(load_datas)

        return load_data

class Storage_gen(object):

    def __init__(self,low_w=4000,high_w=5000,dt_time=1):
        """ low_w is lower point of load power
            high_w is maximum point of load power
            dt_time is discretization of 24hour in to that form
        """
        self.low_w=low_w
        self.high_w=high_w
        self.dt_time=dt_time

    def _data(self,_head):
        """create data for load"""
        data={}
        storage_soc=np.random.randint(self.low_w,self.high_w,size=self.dt_time)
        data[_head]=storage_soc
        return pd.DataFrame(data,columns= [_head])

    def _datas(self,cols):
        sto_cols=[]
        for col in range(len(cols)):
            if len(sto_cols)==0:
                sto_data=self._data(cols[col])
                sto_cols.append(sto_data.columns[0])
            else:
                sto_datas=self._data(cols[col])
                if sto_datas.columns[0] not in sto_cols:
                    sto_cols.append(sto_datas.columns[0])
                    sto_data=sto_data.join(sto_datas)

        return sto_data

class Grid_gen(object):

    def __init__(self,pw_type=3,low_w=18000,high_w=20000,dt_time=24):
        """ low_w is lower point of load power
            high_w is maximum point of load power
            dt_time is discretization of 24hour in to that form
        """
        self.low_w=low_w
        self.pw_type=pw_type
        self.high_w=high_w
        self.dt_time=dt_time


    def _data(self,_head):
        """create data for load"""
        data={}
        grid=self._copy_data()
        data[_head]=grid
        return pd.DataFrame(data,columns= [_head])

    def _copy_data(self):
        data=np.random.randint(self.low_w,self.high_w,size=self.pw_type)
        data.sort()
        datas=[20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,20000,]
        # for i in range(self.pw_type):
        #     re_data=list([data[i]]*(self.dt_time//self.pw_type))
        #     for j in range(len(re_data)):
        #         datas.append(re_data[j])
        return datas

    def _datas(self,cols):
        grid_cols=[]
        for col in range(len(cols)):
            if len(grid_cols)==0:
                grid_data=self._data(cols[col])
                grid_cols.append(grid_data.columns[0])
            else:
                grid_datas=self._data(cols[col])
                if grid_datas.columns[0] not in grid_cols:
                    grid_cols.append(grid_datas.columns[0])
                    grid_data=grid_data.join(grid_datas)

        return grid_data

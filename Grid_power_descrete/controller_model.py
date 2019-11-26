import numpy as np 

def balance(self,storage_max,storage_min,soc,pv,load,action,hour,dt=1):
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
        if (storage_max - soc) > 2000*dt:
            storage_need=2000*dt
        else:
            storage_need=2000*dt

        if soc >storage_min:
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
                st_2ld   =load-(pv_nfill*dt)
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

        return  grid_buy,grid_sell,pv_2sell,pv_2st,pv_2ld,st_2ld,st_4grid,st_4pv,load_4grid

import openseespy.opensees as ops
from libdenavit.section.wide_flange import *
from libdenavit.OpenSees import AnalysisResults
from Units import WF_Database  
import opsvis as opsv
from libdenavit.OpenSees.plotting import *

class Stepped_Column:

    def __init__(self,bottom_column_section_name,height_of_bottom_column,number_of_elements_bottom_column,load_on_bottom_column,
                 top_column_section_name,height_of_top_column,number_of_elements_top_column,load_on_top_column,offset_1,offset_2,offset_3,**kwargs):

        self.height_of_bottom_column = height_of_bottom_column
        self.height_of_top_column = height_of_top_column
        self.number_of_elements_bottom_column = number_of_elements_bottom_column
        self.number_of_elements_top_column = number_of_elements_top_column
        self.load_on_bottom_column = load_on_bottom_column
        self.load_on_top_column = load_on_top_column
        self.bottom_column_section_name = bottom_column_section_name
        self.top_column_section_name = top_column_section_name 
        self.offset1 = offset_1  ## dist from top of bottom column to right node where point load is applied for bottom column
        self.offset2 = offset_2  ## dist from top of bottom column to left node ie bottom of top column
        self.offset3 = offset_3  ## dist from top of top column to right node where load is applied for top column
        self.length_of_single_bottom_column_element = height_of_bottom_column/number_of_elements_bottom_column 
        self.length_of_single_top_column_element = height_of_top_column/number_of_elements_top_column

        defaults={'nip':3,
                  'mat_type':'Steel01',
                  'nfy':20,
                  'nfx':20,
                  'num_regions':10,
                  'Fy':36,
                  'E':29000,
                  'frc':0.0,
                  'Elastic_analysis':False,
                  'Second_order_effects':True,
                  'Residual_Stress':True,
                  'Geometric_Imperfection':False,
                  'geometric_imperfection_ratio':1/500,
                  'stiffness_reduction':1,
                  'strength_reduction':1,
                  'show_model':True}    
        
        for key,value in defaults.items():
            setattr(self,key,kwargs.get(key,value))


    def build_stepped_column(self):

        ops.wipe()
        ops.model('basic','-ndm',2,'-ndf',3)

        ## Bottom node of bottom column
        ops.node(1,0.0,0.0)

        ## Create nodes along bottom column
        for i in range(self.number_of_elements_bottom_column):
            x=0.0
            y=(i+1)*self.length_of_single_bottom_column_element
            if not self.Geometric_Imperfection:
                ops.node(i+2,x,y)
            else:
                ops.node(i+2,x+self.geometric_imperfection_ratio*y,y)
        bottom_column_top_node_tag=self.number_of_elements_bottom_column+1

        ## Create right offset node for bottom column
        right_offset1_node_tag=bottom_column_top_node_tag+1
        x=self.offset1
        y=self.height_of_bottom_column
        if not self.Geometric_Imperfection:
            ops.node(right_offset1_node_tag,x,y)
        else:
            ops.node(right_offset1_node_tag,x+self.geometric_imperfection_ratio*y,y)
        
        ## Create nodes along top column
        top_column_bottom_node_tag=right_offset1_node_tag+1
        if not self.Geometric_Imperfection:
            ops.node(top_column_bottom_node_tag,-self.offset2,self.height_of_bottom_column)
        else:
            ops.node(top_column_bottom_node_tag,-self.offset2+self.geometric_imperfection_ratio*self.height_of_bottom_column,self.height_of_bottom_column)

        for j in range(self.number_of_elements_top_column):
            if not self.Geometric_Imperfection:
                ops.node(top_column_bottom_node_tag+j+1,-self.offset2,self.height_of_bottom_column+(j+1)*self.length_of_single_top_column_element)
            else:
                 ops.node(top_column_bottom_node_tag+j+1,-self.offset2+self.geometric_imperfection_ratio*(self.height_of_bottom_column+(j+1)*self.length_of_single_top_column_element),self.height_of_bottom_column+(j+1)*self.length_of_single_top_column_element)
            
        top_column_top_node_tag=top_column_bottom_node_tag+self.number_of_elements_top_column

        ## Create right offset node for top column
        right_offset3_node_tag=top_column_top_node_tag+1
        x=self.offset3
        y=self.height_of_bottom_column+self.height_of_top_column
        if not self.Geometric_Imperfection:
            ops.node(right_offset3_node_tag,x,y)
        else:
            ops.node(right_offset3_node_tag,x+self.geometric_imperfection_ratio*y,y)
        

        ## Define geometric transformation for columns
        col_TransTag=1
        if self.Second_order_effects:    
            ops.geomTransf("PDelta", col_TransTag)

        else:
            ops.geomTransf("Linear", col_TransTag)

        if self.Elastic_analysis:
            mat_type = 'Elastic'
            frc = 0
        else:
            mat_type = self.mat_type
            frc = -0.3 * self.Fy if self.Residual_Stress else 0

        ## Fix base of bottom column
        ops.fix(1,1,1,0)

        ## Define bottom column section 
        bottom_column_section_tag=1
        bottom_column_section=WF_Database(self.bottom_column_section_name,unit=1)
        bottom_column = I_shape(bottom_column_section.d, bottom_column_section.tw, bottom_column_section.bf, bottom_column_section.tf,   
            self.Fy,self.E,
            A=bottom_column_section.A, 
            Ix=bottom_column_section.Ix,Zx=bottom_column_section.Zx,Sx=bottom_column_section.Sx,rx=bottom_column_section.rx,
            Iy=bottom_column_section.Iy,Zy=bottom_column_section.Zy,Sy=bottom_column_section.Sy,ry=bottom_column_section.ry,
            J=bottom_column_section.J,Cw=bottom_column_section.Cw,rts=bottom_column_section.rts,ho=bottom_column_section.ho)
        bottom_column.build_ops_fiber_section(bottom_column_section_tag,
                                    start_material_id=1,
                                    mat_type=mat_type,
                                    nfy=self.nfy, nfx=self.nfx,
                                    frc=frc,num_regions=self.num_regions,
                                    stiffness_reduction=self.stiffness_reduction,strength_reduction=self.strength_reduction,
                                    axis='x')
        
        ## Define bottom column elements
        ops.beamIntegration("Lobatto", bottom_column_section_tag, bottom_column_section_tag, self.nip)
        for k in range(self.number_of_elements_bottom_column):
            element_tag=k+1
            node_i_tag=k+1
            node_j_tag=k+2
            ops.element('forceBeamColumn',element_tag,node_i_tag,node_j_tag,
                        col_TransTag,
                        bottom_column_section_tag)
        
        ## Define offset beam section
        offset_beam_section_tag=2
        ops.section('Elastic', offset_beam_section_tag, 2.0e11, 1.0e6, 1.0e6)
        ## Define beam integration for offset beam section
        ops.beamIntegration("Lobatto", offset_beam_section_tag, offset_beam_section_tag, self.nip)
        ## Define offset beam element at top of bottom column
        offset1_element_tag=self.number_of_elements_bottom_column+1
        ops.element('forceBeamColumn',offset1_element_tag,bottom_column_top_node_tag,right_offset1_node_tag,
                    col_TransTag,offset_beam_section_tag)
        
        ## Define 2nd offset beam element at bottom of top column
        offset2_element_tag=offset1_element_tag+1
        ops.element('forceBeamColumn',offset2_element_tag,bottom_column_top_node_tag,top_column_bottom_node_tag,
                    col_TransTag,offset_beam_section_tag)
        
        ## Define top column section
        top_column_section_tag=3
        top_column_section=WF_Database(self.top_column_section_name,unit=1)
        top_column = I_shape(top_column_section.d, top_column_section.tw, top_column_section.bf, top_column_section.tf,   
            self.Fy,self.E,
            A=top_column_section.A, 
            Ix=top_column_section.Ix,Zx=top_column_section.Zx,Sx=top_column_section.Sx,rx=top_column_section.rx,
            Iy=top_column_section.Iy,Zy=top_column_section.Zy,Sy=top_column_section.Sy,ry=top_column_section.ry,
            J=top_column_section.J,Cw=top_column_section.Cw,rts=top_column_section.rts,ho=top_column_section.ho)
        top_column.build_ops_fiber_section(top_column_section_tag,
                                    start_material_id=200,
                                    mat_type=mat_type,
                                    nfy=self.nfy, nfx=self.nfx,
                                    frc=frc,num_regions=self.num_regions,
                                    stiffness_reduction=self.stiffness_reduction,strength_reduction=self.strength_reduction,
                                    axis='x')
        ## Define beam integration for top column section
        ops.beamIntegration("Lobatto", top_column_section_tag, top_column_section_tag, self.nip)       
        ## Define top column elements
        for m in range(self.number_of_elements_top_column):
            element_tag=self.number_of_elements_bottom_column+3+m
            node_i_tag=top_column_bottom_node_tag+m
            node_j_tag=top_column_bottom_node_tag+m+1
            ops.element('forceBeamColumn',element_tag,node_i_tag,node_j_tag,
                        col_TransTag,
                        top_column_section_tag)
            
        ## Define offset beam element at top of top column
        offset3_element_tag=self.number_of_elements_bottom_column+3+self.number_of_elements_top_column 
        ops.element('forceBeamColumn',offset3_element_tag,top_column_top_node_tag,right_offset3_node_tag,
                    col_TransTag,offset_beam_section_tag)
        
        ## Support at top node of top column 
        ops.fix(top_column_top_node_tag,1,0,0)

        ## Apply point load at right offset node of bottom column and offset node of top column
        ops.timeSeries('Linear',1)
        ops.pattern('Plain',1,1)
        ops.load(right_offset1_node_tag,0.0,-self.load_on_bottom_column,0.0)  ## Apply vertical downward load
        ops.load(right_offset3_node_tag,0.0,-self.load_on_top_column,0.0)  ## Apply vertical downward load

        if self.show_model: 
            opsv.plot_model()
            opsv.plot_load()
    
    
    def plot_model(self):
        # show quick model diagnostics then plot using libdenavit's plotting
        try:
            node_coords = get_node_coords()
            element_nodes = get_element_nodes()
            print(f"Plotting model: {len(node_coords)} nodes, {len(element_nodes)} elements")
        except Exception:
            print("Unable to read node/element info for diagnostics")
        plot_undeformed_2d(axis_equal=True)


Stepped_Column = Stepped_Column(bottom_column_section_name='W14X90',height_of_bottom_column=120.0,number_of_elements_bottom_column=1,load_on_bottom_column=10,
                                    top_column_section_name='W14X132',height_of_top_column=80.0,number_of_elements_top_column=1,load_on_top_column=5,
                                    offset_1=4.0,offset_2=4.0,offset_3=6.0,
                                    Fy=36,E=29000,
                                    Elastic_analysis=False,
                                    Second_order_effects=True,
                                    Residual_Stress=True,
                                    Geometric_Imperfection=True,
                                    geometric_imperfection_ratio=-1/10)

Stepped_Column.build_stepped_column()
Stepped_Column.plot_model()

"""
Creating tikz figure using python code (https://github.com/HarisIqbal88/PlotNeuralNet)
"""

import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [ 
    #to_head('..'), 
    #to_cor(),
    #to_begin(),
    r"""\begin{tikzpicture}""",

    r"""\pic[shift={ (0,0,0) }] at (-3,0,0) 
    {RightBandedBox={
        name=ccr_b0,
        caption=Input (128x128),
        fill=white,
        bandfill=black,
        height=40,
        width=1,
        depth=40
        }
    };""",

     #input
    to_input( 'images/GT_phantom.png' ),

    #encoder
    # deep1
    to_ConvConvRelu( name='ccr_b1', s_filer=128, n_filer=(16,16), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40  ),
    # down1
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=2, height=32, depth=32, opacity=0.5),
    
    # deep2
    to_ConvConvRelu( name='ccr_b2', s_filer=64, n_filer=(32,32), offset="(1,0,0)", to="(pool_b1-east)", width=(3.5,3.5), height=32, depth=32 ),
    # down2
    to_Pool(name="pool_b2", offset="(0,0,0)", to="(ccr_b2-east)", width=3.5, height=25, depth=25, opacity=0.5),
    to_connection("pool_b1","ccr_b2"),
    
    # deep3
    to_ConvConvRelu( name='ccr_b3', s_filer=32, n_filer=(64,64), offset="(1,0,0)", to="(pool_b2-east)", width=(4.5,4.5), height=25, depth=25 ),
    # down3
    to_Pool(name="pool_b3", offset="(0,0,0)", to="(ccr_b3-east)", width=4.5, height=16, depth=16, opacity=0.5),
    to_connection("pool_b2","ccr_b3"),
    
    #Decoder
    # deep4
    to_ConvConvRelu( name='ccr_b6', s_filer=16, n_filer=(128,128), offset="(2,0,0)", to="(pool_b3-east)", width=(6,6), height=16, depth=16 ),
    # up1
    to_UnPool(name="unpool_b1", offset="(0,0,0)", to="(ccr_b6-east)", width=6, height=25, depth=25, opacity=0.5),
    to_Conv(name='ccr2_b6', offset="(0,0,0)", to="(unpool_b1-east)", s_filer=32, n_filer=64, width=4.5, height=25, depth=25 ),       
    to_connection( "pool_b3", "ccr_b6"),
    to_skip( of='ccr_b3', to='ccr2_b6', pos=1.25),    

    # deep5
    to_ConvConvRelu( name='ccr_b7', s_filer=32, n_filer=(64,64), offset="(2,0,0)", to="(unpool_b1-east)", width=(4.5,4.5), height=25, depth=25 ),
    # up2
    to_UnPool(name="unpool_b2", offset="(0,0,0)", to="(ccr_b7-east)", width=4.5, height=32, depth=32, opacity=0.5),
    to_Conv(name='ccr2_b7', offset="(0,0,0)", to="(unpool_b2-east)", s_filer=64, n_filer=32, width=3.5, height=32, depth=32 ),       
    to_connection("ccr2_b6","ccr_b7"),
    to_skip( of='ccr_b2', to='ccr2_b7', pos=1.25),    

    # deep6
    to_ConvConvRelu( name='ccr_b8', s_filer=64, n_filer=(32,32), offset="(2,0,0)", to="(unpool_b2-east)", width=(3.5,3.5), height=32, depth=32 ),
    # up3
    to_UnPool(name="unpool_b3", offset="(0,0,0)", to="(ccr_b8-east)", width=3.5, height=40, depth=40, opacity=0.5),
    to_Conv(name='ccr2_b8', offset="(0,0,0)", to="(unpool_b3-east)", s_filer=128, n_filer=16, width=2, height=40, depth=40 ),       
    to_connection( "ccr2_b7", "ccr_b8"),
    to_skip( of='ccr_b1', to='ccr2_b8', pos=1.25),    
    
    # deep7
    to_ConvConvRelu( name='ccr_b9', s_filer='', n_filer=(16,1), offset="(2,0,0)", to="(unpool_b3-east)", width=(2,1), height=40, depth=40  ),
    # positivity
    to_ConvSoftMax( name="soft1", s_filer=128, offset="(0,0,0)", to="(ccr_b9-east)", width=1, height=40, depth=40, caption="Output (128x128)" ),
    to_connection( "ccr2_b8", "ccr_b9"),
    
    #to_end()

    r"""%%%%%%%%%%%%%% title and legend %%%%%%%%%%%%%%%
    \node[above,font=\large\bfseries] at (current bounding box.north) {Variational AutoEncoder (DIP based architecture)};

    \matrix [draw,below left] at (current bounding box.south east) {
    \node [convStyle] {}; \\
    \node [downStyle] {}; \\
    \node [linearStyle] {}; \\
    \node [upStyle] {}; \\
    \node [skipStyle] {}; \\
    \node [outputStyle] {}; \\
    };
    """,
    
    r"""\end{tikzpicture}"""
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    

    

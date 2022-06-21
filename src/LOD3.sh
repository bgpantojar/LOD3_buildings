#!/bin/bash
#FreeCAD path
export FREECADPATH=/home/pantojas/Bryan_PR2/04_TAPEDA_BP/04_UAV2EFM/08_LOD3_generation/LOD3_buildings/freecad_dev/usr/bin
#LOD3_buildings path
export LOD3PATH=/home/pantojas/Bryan_PR2/04_TAPEDA_BP/04_UAV2EFM/08_LOD3_generation/LOD3_buildings/src

$main_op_file=$1
$main_LOD3_file=$2

PYTHONPATH=${LOD3PATH} PATH=$PATH:${FREECADPATH} python $main_op_file $@

PYTHONPATH=${LODPATH} PATH=$PATH:${FREECADPATH} freecadcmd $main_LOD3_file $@

#From terminal, being inside src folder, activate environment and run in terminal as:  
#./LOD3.sh ../examples/p2_LOD3_00_School_main_op.py ../examples/p2_LOD3_00_School_main_LOD3.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:44:32 2020

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430


@author: pantoja
"""

import numpy as np
#The next modules are comming from freecad commandline 
import Part
import FreeCAD
from FreeCAD import Base
import importOBJ

#CLASES

class Building:
    '''
    Building class
    nf: number of faces (int)
    faces: list with Face objects 
    opens: list with opens objects 
    assemble_b: method to assemble building faces
    assemble_o: method to assemble openings faces
    assemble_bo: method to assemble building with openings
    assemble_boc: method to assemble building with openings and cracks
    '''
    def __init__(self, nf, faces=None, opens=None):
        self.nf = nf
        self.faces = faces
        self.opens= opens

    def assemble_b(self):
        for i in enumerate(self.faces):
            myshell = Part.makeShell([self.faces[i].face_free for i in range(len(self.faces))])
            Part.show(myshell)
            return myshell

    def assemble_o(self):
        for i in enumerate(self.opens):
            myshell = Part.makeShell([self.opens[i].open_free for i in range(len(self.opens))])
            Part.show(myshell)
            return myshell

    def assemble_bo(self):
        for i,f in enumerate(self.faces):
            if i==0:
                myshell_f = self.faces[i].face_free.fuse(self.faces[i+1].face_free)
            elif i==len(self.faces)-1:
                break
            else:
                myshell_f = myshell_f.fuse(self.faces[i+1].face_free)
        
        for i,o in enumerate(self.opens):
            if i==0:
                myshell_o = self.opens[i].open_free_ext.fuse(self.opens[i+1].open_free_ext) 
            elif i==len(self.opens)-1:
                break
            else:
                myshell_o = myshell_o.fuse(self.opens[i+1].open_free_ext)

        myshell = myshell_f.cut(myshell_o)
        Part.show(myshell)
        return myshell
    

class Face:
    '''
    Face class
    f_vert: list with face vertices coordinates
    th: face thickness (int)
    buid_face: method to build face with its vertices information
    face_free: Freecad face entity
    opens_free: List of opens as Freecad face entities 
    '''
    def __init__(self, f_vert={}, th=0, face_free="NaN"):
        self.f_vert = f_vert
        self.th = th
        self.face_free = face_free
    
    def build_face(self):
        poly = Part.makePolygon([tuple(self.f_vert[i]) for i in range(len(self.f_vert))])
        face = Part.Face(poly)
        return face

class Open:
    '''
    Open class
    o_vert: list with the open vertices coordinates
    open_free = List of opens as Freecad face entities
    open_free_ext = List of opens as Freecad extruded face entities
    '''
    def __init__(self, o_vert={}, open_free="NaN", open_free_ext="NaN"):
        self.o_vert = o_vert
        self.open_free = open_free
        
    def build_open(self):
        poly = Part.makePolygon([tuple(self.o_vert[i]) for i in range(len(self.o_vert))])
        openn = Part.Face(poly)
        return openn
    
    def build_open_ext(self):
        poly = Part.makePolygon([tuple(self.o_vert[i]) for i in range(len(self.o_vert))])
        openn = Part.Face(poly)
        #Finding normal to triangular opening to extrude in that direction
        P = [np.array(tuple(self.o_vert[i])) for i in range(len(self.o_vert)-1)]
        normal = np.cross(P[0]-P[1], P[0]-P[2])
        normal = normal * (1/np.linalg.norm(normal))
        lenght = min(np.linalg.norm(P[0]-P[1]),np.linalg.norm(P[0]-P[2]),np.linalg.norm(P[1]-P[2]))
        openn_e1 = openn.extrude(Base.Vector(3*lenght*normal)) #!
        openn_e2 = openn.extrude(Base.Vector(-3*lenght*normal)) #!
        openn_e = openn_e1.fuse(openn_e2)
        
        return openn_e


#BULDING THE LOD3 model using structures and methods
def op_builder(data_folder, data_path, polyfit_path, dense=False):
    #Loading polyfit model vertices and faces
    vertices_p = list()
    faces_p = list()
    if dense:
        f = open(polyfit_path +"/polyfit_dense.obj", "r")
    else:
        f = open(polyfit_path +"/polyfit.obj", "r")
        
    for i in f:
        l = i.split()
        if l[0] == 'v':
            vertices_p.append([float(j) for j in l[1:]])
        else:
            faces_p.append([int(j) for j in l[1:]])
    f.close()

    #Loading .obj openigs
    vertices_o = list()
    openings_o = list()

    f = open(data_path + "/openings.obj", "r")
    for i in f:
        l = i.split()
        if l[0] == 'v':
            vertices_o.append([float(j) for j in l[1:]])
        else:
            openings_o.append([int(j) for j in l[1:]])
    f.close()
    
    ### Begin command Std_New

    doc_name = data_folder
    th = 2. 
    
    faces = list({})
    for i, f in enumerate(faces_p):
        F = Face()
        F.th = th
        F.f_vert = list({})
        
        for j in f:
            F.f_vert.append(vertices_p[j-1])
        F.f_vert.append(vertices_p[f[0]-1])
        F.face_free = F.build_face()
        faces.append(F)

    opens = list({})
    for i, o in enumerate(openings_o):
        O = Open()
        O.o_vert = list({})
        for j in o:
            O.o_vert.append(vertices_o[j-1])
        O.o_vert.append(vertices_o[o[0]-1])
        
        O.open_free = O.build_open()
        O.open_free_ext = O.build_open_ext()
        opens.append(O)

    B = Building(len(faces), faces, opens)
    B.assemble_bo()

    #Saving as .obj model
    objs=[]
    objs.append(FreeCAD.getDocument("Unnamed").getObject("Shape"))
    FreeCAD.getDocument("Unnamed").saveAs(u"../results/"+data_folder+"/LOD3.FCStd")
    importOBJ.export(objs,u"../results/"+data_folder+"/LOD3.obj")
    del objs
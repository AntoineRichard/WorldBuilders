from pxr import UsdGeom, Gf, Sdf, UsdPhysics, UsdShade, Usd, Vt
from typing import Tuple, Union, List
from omni.physx.scripts import utils
import numpy as np
import omni
import os

def loadStage(path: str,
              ) -> None:
    """
    Loads the stage from the given path.
    
    Args:
        path (str): The path to the stage."""

    omni.usd.get_context().open_stage(path)

def saveStage(path: str,
              ) -> None:
    """
    Saves the stage to the given path.
    
    Args:
        path (str): The path to the stage."""
    
    omni.usd.get_context().save_as_stage(path, None)

def newStage() -> None:
    """
    Creates a new stage.
    
    Args:
        path (str): The path to the stage."""
    
    omni.usd.get_context().new_stage()

def closeStage() -> None:
    """
    Closes the current stage."""

    omni.usd.get_context().close_stage()

def setDefaultPrim(stage: Usd.Stage,
                   path:str,
                   ) -> None:
    """
    Sets the default prim of the stage.
    
    Args:
        stage (Usd.Stage): The stage.
        path (str): The path to the default prim."""
    
    prim = stage.GetPrimAtPath(path)
    stage.SetDefaultPrim(prim)

def movePrim(path_from:str,
             path_to:str,
             ) -> None:
    """
    Moves a prim from one location to another.
    There has to be some native PXR function for this, but I couldn't find it.
    
    Args:
        path_from (str): The path to the prim to move.
        path_to (str): The path to the new location of the prim."""

    omni.kit.commands.execute('MovePrim',path_from=path_from, path_to=path_to)

def createXform(stage: Usd.Stage,
                path:str,
                add_default_op:bool=False,
                ) -> Tuple[Usd.Prim, str]:
    """
    Creates a Xform prim and adds it to the stage.
    
    Args:
        stage (Usd.Stage): The stage.
        path (str): The path to the prim.
        add_default_op (bool, optional): Whether to add the default ops to the prim. Defaults to False.
        
    Returns:
        Tuple[Usd.Prim, str]: The prim and its path."""
    
    prim_path = omni.usd.get_stage_next_free_path(stage, path, False)
    obj_prim = stage.DefinePrim(prim_path, "Xform")

    if add_default_op:
        addDefaultOps(obj_prim)
    return obj_prim, prim_path

def addDefaultOps(prim: Usd.Prim,
                  ) -> UsdGeom.Xform:
    """
    Adds the default ops to a prim.
    
    Args:
        prim (Usd.Prim): The prim.
    
    Returns:
        UsdGeom.Xform: The Xform of the prim."""
    xform = UsdGeom.Xform(prim)
    xform.ClearXformOpOrder()

    try:
        xform.AddTranslateOp(precision = UsdGeom.XformOp.PrecisionDouble)
    except:
        pass
    try:
        xform.AddOrientOp(precision = UsdGeom.XformOp.PrecisionDouble)
    except:
        pass
    try:
        xform.AddScaleOp(precision = UsdGeom.XformOp.PrecisionDouble)
    except:
        pass
    return xform

def setDefaultOps(xform: UsdGeom.Xform,
                  pos: Tuple[float, float, float],
                  rot: Tuple[float, float, float, float],
                  scale: Tuple[float, float, float],
                  ) -> None:
    """
    Sets the default ops of a Xform prim using lists instead of the native values (Gf.Vec3d, Gf.Vec3f, Gf.Quatd, Gf.Quatf).
    
    Args:
        xform (UsdGeom.Xform): The Xform prim.
        pos (tuple): The position.
        rot (tuple): The rotation as a quaternion.
        scale (tuple): The scale."""
    
    xform = UsdGeom.Xform(xform)
    xform_ops = xform.GetOrderedXformOps()
    try:
        xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    except:
        xform_ops[0].Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
    try:
        xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])))
    except:
        xform_ops[1].Set(Gf.Quatd(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])))
    try:
        xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
    except:
        xform_ops[2].Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))

def setDefaultOpsTyped(xform: UsdGeom.Xform,
                       pos: Union[Gf.Vec3d, Gf.Vec3f],
                       rot: Union[Gf.Quatd, Gf.Quatf],
                       scale: Union[Gf.Vec3d, Gf.Vec3f],
                       ) -> None:
    """
    Sets the default ops of a Xform prim using the native values (Gf.Vec3d, Gf.Vec3f, Gf.Quatd, Gf.Quatf).
    
    Args:
        xform (UsdGeom.Xform): The Xform prim.
        pos (Union[Gf.Vec3d, Gf.Vec3f]): The position.
        rot (Union[Gf.Quatd, Gf.Quatf]): The rotation as a quaternion.
        scale (Union[Gf.Vec3d, Gf.Vec3f]): The scale."""

    xform_ops = xform.GetOrderedXformOps()
    xform_ops[0].Set(pos)
    xform_ops[1].Set(rot)
    xform_ops[2].Set(scale)

def loadTexture(stage: Usd.Stage,
                mdl_path: str,
                scene_path: str,
                sub_intentifier: str,
                ) -> None:
    """
    Loads a texture from a MDL file and adds it to the stage.
    
    Args:
        stage (Usd.Stage): The stage.
        mdl_path (str): The path to the MDL file.
        scene_path (str): The path to the scene.
        sub_intentifier (str): The sub identifier of the texture."""
    
    mtl_path = Sdf.Path(scene_path)
    mtl = UsdShade.Material.Define(stage, mtl_path)
    shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    # MDL shaders should use "mdl" sourceType
    shader.SetSourceAsset(mdl_path, "mdl")
    shader.SetSourceAssetSubIdentifier(sub_intentifier, "mdl")
    # MDL materials should use "mdl" renderContext
    mtl.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mtl.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mtl.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mtl_prim = stage.GetPrimAtPath(mtl_path)
    material = UsdShade.Material(mtl_prim)
    return material

def applyMaterial(prim: Usd.Prim,
                  material: UsdShade.Material,
                  ) -> None:
    """
    Applies a material to a prim.
    
    Args:
        prim (Usd.Prim): The prim.
        material (UsdShade.Material): The material."""

    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

def applyMaterialFromPath(stage: Usd.Stage,
                          prim_path: str,
                          material_path: str,
                          ) -> None:
    """
    Applies a material to a prim.
    
    Args:
        stage (Usd.Stage): The stage.
        prim_path (str): The path to the prim.
        material_path (str): The path to the material."""
    
    mtl_prim = stage.GetPrimAtPath(material_path)
    prim = stage.GetPrimAtPath(prim_path)
    material = UsdShade.Material(mtl_prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

def createObject(prefix: str,
    stage: Usd.Stage,
    path: str,
    position: Gf.Vec3d = Gf.Vec3d(0, 0, 0),
    rotation: Gf.Quatd = Gf.Quatd(0,0,0,1),
    scale: Gf.Vec3d = Gf.Vec3d(1,1,1),
    is_instance: bool = True,
    ) -> Tuple[Usd.Prim, str]:
    """
    Creates a 3D object from a USD file and adds it to the stage.
    
    Args:
        prefix (str): The prefix of the object.
        stage (Usd.Stage): The stage.
        path (str): The path to the USD file.
        position (Gf.Vec3d, optional): The position of the object. Defaults to Gf.Vec3d(0, 0, 0).
        rotation (Gf.Quatd, optional): The rotation of the object. Defaults to Gf.Quatd(0,0,0,1).
        scale (Gf.Vec3d, optional): The scale of the object. Defaults to Gf.Vec3d(1,1,1).
        is_instance (bool, optional): Whether the object is an instance or not. Defaults to True."""
    
    obj_prim, prim_path = createXform(stage, prefix)
    xform = UsdGeom.Xformable(obj_prim)
    addDefaultOps(xform)
    setDefaultOpsTyped(xform, position, rotation, scale)
    obj_prim.GetReferences().AddReference(path)
    if is_instance:
        obj_prim.SetInstanceable(True)
    return obj_prim, prim_path

def addCollision(stage: Usd.Stage,
                 path: str,
                 mode: str = "none",
                 ) -> None:
    """
    Adds a collision to a prim.
    
    Args:
        stage (Usd.Stage): The stage.
        path (str): The path to the prim.
        mode (str, optional): The mode of the collision. Defaults to "none"."""

    # Checks that the mode selected by the user is correct.
    accepted_modes = ["none", "convexHull", "convexDecomposition", "meshSimplification", "boundingSphere", "boundingCube"]
    assert mode in accepted_modes, "Decimation mode: "+mode+" for colliders unknown."
    # Get the prim and add collisions.
    prim = stage.GetPrimAtPath(path)
    utils.setCollider(prim, approximationShape=mode)

def removeCollision(stage: Usd.Stage,
                    path: str,
                    ) -> None:
    """
    Removes a collision from a prim.
    
    Args:
        stage (Usd.Stage): The stage.
        path (str): The path to the prim."""
    
    # Get the prim and remove collisions.
    prim = stage.GetPrimAtPath(path)
    utils.removeCollider(prim)

def deletePrim(stage: Usd.Stage,
               path: str,
               ) -> None:
    """
    Deletes a prim from the stage.
    
    Args:
        stage (Usd.Stage): The stage.
        path (str): The path to the prim."""
    
    # Deletes a prim from the stage.
    stage.RemovePrim(path)

def createStandaloneInstance(stage: Usd.Stage,
                             path: str,
                             ) -> UsdGeom.PointInstancer:
    """
    Creates only the instance object.
    
    Args:
        stage (Usd.Stage): The stage.
        path (str): The path to the instance.
    
    Returns:
        UsdGeom.PointInstancer: The instance."""
    
    # Creates and instancer.
    createXform(stage, path, add_default_op=True)
    instancer = UsdGeom.PointInstancer.Define(stage, path)
    return instancer

def createInstancerAndCache(stage: Usd.Stage,
                            path: str,
                            asset_list: List[str],
                            ) -> None:
    """
    Creates an instancer and its cache.
    
    Args:
        stage (Usd.Stage): The stage.
        path (str): The path to the instancer.
        asset_list (List[str]): The list of assets to cache."""
    
    # Creates a point instancer
    instancer = createStandaloneInstance(stage, path)
    # Creates a Xform to cache the assets to.
    # This cache must be located under the instancer to hide the cached assets.
    #createXform(stage, os.path.join(path,'cache'))
    # Add each asset to the scene in the cache.
    for asset in asset_list:
        # Create asset.
        #prim, prim_path = createObject(os.path.join(path,'cache','instance'), stage, asset)
        prim, prim_path = createObject(os.path.join(path,'instance'), stage, asset)
        #prim_sd = PrimSemanticData(prim)
        #prim_sd.add_entry("class", "rock")
        # Add this asset to the list of instantiable objects.
        instancer.GetPrototypesRel().AddTarget(prim_path)
    # Set some dummy parameters
    setInstancerParameters(stage, path, pos=np.zeros((1,3))) 

def createInstancerFromCache(stage: Usd.Stage,
                             path: str,
                             cache_path: str,
                             ) -> None:
    """
    Creates an instancer from a cache.
    
    Args:
        stage (Usd.Stage): The stage.
        path (str): The path to the instancer.
        cache_path (str): The path to the cache."""
    
    # Creates a point instancer
    instancer = createStandaloneInstance(stage, path)
    # Add each asset to the scene in the cache.
    for asset in Usd.PrimRange(stage.GetPrimAtPath(cache_path)):
        instancer.GetPrototypesRel().AddTarget(asset.GetPath())
    # Set some dummy parameters
    setInstancerParameters(stage, path, pos=np.zeros((1,3))) 

def setInstancerParameters(stage: Usd.Stage,
                           path: str,
                           pos: np.ndarray([],dtype=float),
                           ids: np.ndarray([],dtype=int) = None,
                           scale: np.ndarray([],dtype=float) = None,
                           quat: np.ndarray([],dtype=float) = None,
                           ) -> None:
    """
    Sets the parameters of an instancer.
    
    Args:
        stage (Usd.Stage): The stage.
        path (str): The path to the instancer.
        pos (np.ndarray): The positions of the instances.
        ids (np.ndarray, optional): The ids of the instances. Defaults to None.
        scale (np.ndarray, optional): The scale of the instances. Defaults to None.
        quat (np.ndarray, optional): The orientation of the instances. Defaults to None."""

    num = pos.shape[0]
    instancer_prim = stage.GetPrimAtPath(path)
    num_prototypes = len(instancer_prim.GetRelationship("prototypes").GetTargets())
    # Set positions.
    instancer_prim.GetAttribute("positions").Set(pos)
    # Set scale.
    if scale is None:
        scale = np.ones_like(pos)
    instancer_prim.GetAttribute("scales").Set(scale)
    # Set orientation.
    if quat is None:
        quat = np.zeros((pos.shape[0],4))
        quat[:,-1] = 1
    instancer_prim.GetAttribute("orientations").Set(quat)
    # Set ids.
    if ids is None:
        ids=  (np.random.rand(num) * num_prototypes).astype(int)
    # Compute extent.
    instancer_prim.GetAttribute("protoIndices").Set(ids)
    updateExtent(stage, path)
    
def updateExtent(stage: Usd.Stage,
                 instancer_path: str,
                 ) -> None:
    """
    Updates the extent of an instancer.
    
    Args:
        stage (Usd.Stage): The stage.
        instancer_path (str): The path to the instancer."""
    
    # Get the point instancer.
    instancer = UsdGeom.PointInstancer.Get(stage, instancer_path)
    # Compute the extent of the objetcs.
    extent = instancer.ComputeExtentAtTime(Usd.TimeCode(0), Usd.TimeCode(0))
    # Applies the extent to the instancer.
    instancer.CreateExtentAttr(Vt.Vec3fArray([
        Gf.Vec3f(extent[0]),
        Gf.Vec3f(extent[1]),
    ]))

def enableSmoothShade(prim: Usd.Prim,
                      extra_smooth: bool = False,
                      ) -> None:
    """
    Enables smooth shading on a prim.
    
    Args:
        prim (Usd.Prim): The prim on which to enable smooth shading.
        extra_smooth (bool, optional): Toggles the use of the smooth method instead of catmullClark. Defaults to False."""

    # Sets the subdivision scheme to smooth.
    prim.GetAttribute("subdivisionScheme").Set(UsdGeom.Tokens.catmullClark)
    # Sets the triangle subdivision rule.
    if extra_smooth:
        prim.GetAttribute("triangleSubdivisionRule").Set(UsdGeom.Tokens.smooth)
    else:
        prim.GetAttribute("triangleSubdivisionRule").Set(UsdGeom.Tokens.catmullClark)

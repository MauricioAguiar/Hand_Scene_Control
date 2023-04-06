import obspython as obs
import cv2
import mediapipe as mp
import numpy as np
import ntpath
import pandas as pd


# Description displayed in the Scripts dialog window
def script_description():
  return """<center><h2>Hand Control!!</h2></center>
            <p>Hand Control it's a plugin that allow you control a scene object with your own hand in the current scene when a hotkey is pressed. Go to <em>Settings
            </em> then <em>Hotkeys</em> to select the key combination.</p><p>Select the font on  'Source Name'</p>
            <p>Make fine adjust on Scale and Position Threshold (These adjusts are necessary due to imperfections on hand trackmotion in some video capture devices)</p>"""

# Global variables to restore the scene item after shake
move_sceneitem = None             # Reference to the modified scene item

move_sceneitem_scale = obs.vec2() # Initial scale of the current on hold scene

# Global variables to hotkeys usage
hotkey_id_array = []
hotkey_names_by_id = {}
open_camera = False

# Global variables holding the values of data settings / properties
source_name = "Image"     # Name of the source to shake
move_scale_threshold = 10 # Moving Scale threshold to avoid animation oscillations from mediapipe detection
move_pos_trheshold = 5    # Moving Position threshold to avoid animation oscillations from mediapipe detection
debug = False

# Data to smooth animation due to mediapipe jitter on detections
smooth_data = {
  "pos_x": 0.5,
  "pos_y": 0.5,
  "scale_x": 0.0,
  "scale_y": 0.0
}

#Basic Startup setting configuration to MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands                    
hands = None
    
#Height and Width from mediapipe pose track screen
height = 1024
width = 768

#Video Capture to track pose
cap = None

# Saves the original rotation angle of the given sceneitem (assumed not None)
def save_sceneitem_for_shake(sceneitem):
  global move_sceneitem_scale, move_scene_handler

  obs.obs_sceneitem_get_scale(sceneitem, move_sceneitem_scale)

  scene_as_source = obs.obs_scene_get_source(obs.obs_sceneitem_get_scene(sceneitem))
  move_scene_handler = obs.obs_source_get_signal_handler(scene_as_source)
  obs.signal_handler_connect(move_scene_handler, "item_remove", on_move_sceneitem_removed)

# Restores the original rotation angle on the scene item after shake
def restore_sceneitem_after_move():
  global move_sceneitem, move_sceneitem_scale
  if move_sceneitem:
    obs.obs_sceneitem_set_scale(move_sceneitem, move_sceneitem_scale)
    move_sceneitem = None
    obs.signal_handler_disconnect(move_scene_handler, "item_remove", on_move_sceneitem_removed)

# Retrieves the scene item of the given source name in the current scene or None if not found
def get_sceneitem_from_source_name_in_current_scene(name):
  result_sceneitem = None
  current_scene_as_source = obs.obs_frontend_get_current_scene()
  if current_scene_as_source:
    current_scene = obs.obs_scene_from_source(current_scene_as_source)
    result_sceneitem = obs.obs_scene_find_source_recursive(current_scene, name)
    obs.obs_source_release(current_scene_as_source)
  return result_sceneitem

# Animates the scene item corresponding to source_name in the current scene
def move_source():
  move_sceneitem = get_sceneitem_from_source_name_in_current_scene(source_name)
  if move_sceneitem:
    font_id = obs.obs_sceneitem_get_id(move_sceneitem)
    if move_sceneitem and obs.obs_sceneitem_get_id(move_sceneitem) != font_id:
      restore_sceneitem_after_move()
    if not move_sceneitem:
      save_sceneitem_for_shake(move_sceneitem)
    video_capture(move_sceneitem)
  else:
    restore_sceneitem_after_move()

# Called every frame
def script_tick(seconds):
  move_source()
  #video_capture(move_sceneitem)



def script_load(settings):
    print("--- " + ntpath.basename(__file__) + " loaded ---")

    # create Hotkey in global OBS Settings
    hotkey_id_array.append(obs.obs_hotkey_register_frontend("Track pose", "On/Off Track pose", hotkey_1_callback))
    hotkey_names_by_id[hotkey_id_array[len(hotkey_id_array)-1]] = "Track pose"

    # load hotkeys from script save file
    for hotkey_id in hotkey_id_array:
        # get the hotkeys data_array from the script settings (was saved under the hotkeys name)  !! find way to use obs_hotkey_get_name instead of tracking the name manually
        hotkey_data_array_from_settings = obs.obs_data_get_array(settings, hotkey_names_by_id[hotkey_id])
        # load the saved hotkeys data_array to the new created hotkey associated with the "hotkey_id"
        obs.obs_hotkey_load(hotkey_id, hotkey_data_array_from_settings)

        obs.obs_data_array_release(hotkey_data_array_from_settings)
  
def hotkey_1_callback(is_pressed):
    # print(f"-- Shortcut 1 ; Data: {data}")
    if is_pressed:
        switch_bool()
        
# Called before data settings are saved
def script_save(settings):

  restore_sceneitem_after_move()

   # save hotkeys in script properties
  for hotkey_id in hotkey_id_array:
      # save each hotkeys data_array into script settings by the hotkeys name  !! find way to use obs_hotkey_get_name instead of tracking the name manually
      obs.obs_data_set_array(settings, hotkey_names_by_id[hotkey_id], obs.obs_hotkey_save(hotkey_id))

  obs.obs_save_sources()

# Callback for item_remove signal
def on_move_sceneitem_removed(calldata):
  restore_sceneitem_after_move()

move_scene_handler = None # Signal handler of the scene kept to restore

# Called to set default values of data settings
def script_defaults(settings):
  global move_sceneitem
  set_cap(cv2.VideoCapture(3))
  set_camera_open(True)
  switch_bool()
  move_sceneitem = get_sceneitem_from_source_name_in_current_scene(source_name)
  obs.obs_data_set_default_string(settings, "source_name", "")
  obs.obs_data_set_default_double(settings, "move_scale_threshold", 10)
  obs.obs_data_set_default_int(settings, "move_pos_threshold", 5)
  obs.obs_data_set_default_bool(settings, "debug", False)  
  obs.obs_sceneitem_get_scale(move_sceneitem, move_sceneitem_scale)
  


# Called to display the properties GUI
def script_properties():
  
  props = obs.obs_properties_create()
  # Drop-down list of sources
  list_property = obs.obs_properties_add_list(props, "source_name", "Source name",
              obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_STRING)
  
    # Button to refresh the drop-down list
  obs.obs_properties_add_button(props, "button", "Refresh list of sources",
    lambda props,prop: True if populate_list_property_with_source_names(list_property) else True)
  populate_list_property_with_source_names(list_property)
  # obs.obs_properties_add_text(props, "source_name", "Source name", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_float_slider(props, "move_scale_threshold", "Scale Threshold", 1, 100, 0.5)
  obs.obs_properties_add_float_slider(props, "move_pos_threshold", "Move Threshold", 1, 100, 0.5)
  obs.obs_properties_add_bool(props, "debug", "Cam Debug")
  return props

# Called after change of settings including once after script load
def script_update(settings):
  global source_name, move_scale_threshold, move_pos_trheshold, debug, move_sceneitem, move_sceneitem_scale
  restore_sceneitem_after_move()
  source_name = obs.obs_data_get_string(settings, "source_name")
  move_scale_threshold = obs.obs_data_get_double(settings, "move_scale_threshold")
  move_pos_trheshold = obs.obs_data_get_int(settings, "move_pos_threshold")
  debug = obs.obs_data_get_bool(settings, "debug")
  move_sceneitem = get_sceneitem_from_source_name_in_current_scene(source_name)
  obs.obs_sceneitem_get_scale(move_sceneitem, move_sceneitem_scale)
  

# Fills the given list property object with the names of all sources plus an empty one
def populate_list_property_with_source_names(list_property):
  
  sources = obs.obs_enum_sources()
  obs.obs_property_list_clear(list_property)
  obs.obs_property_list_add_string(list_property, "", "")
  for source in sources:
    name = obs.obs_source_get_name(source)
    obs.obs_property_list_add_string(list_property, name, name)
  obs.source_list_release(sources)

def video_capture(sceneitem):
  
    if(is_camera_open()):

      success, image = get_cap().read()
      if not success:
        print ("Video Not Rendered")
      # If loading a video, use 'break' instead of 'continue'.
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      
      else:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = get_hands().process(image)

        try:     
          for hand_landmarks in results.multi_hand_landmarks:             
      
            # Get coordinates
            point_index = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].x,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].y,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].z]

            # Visualize angle
            cv2.putText(image, str(calc_pos(point_index)), 
                tuple(np.multiply(point_index, [height, width]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                )
        except: #Exception as e: print(e.with_traceback)
          pass

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections          
        if results.multi_hand_landmarks:
          obs.obs_sceneitem_set_visible(sceneitem, True)
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        else:
          obs.obs_sceneitem_set_visible(sceneitem, False)

        #if debug:
        cv2.imshow('Tracking Hands Debug', image)

        del image

    else:
      try:
        obs.obs_sceneitem_set_visible(sceneitem, False)
        get_cap().release()
        cv2.destroyAllWindows()
      except:
        pass
        
def set_camera_open(boolean: bool):
  global open_camera
  open_camera = boolean
  
def switch_bool():
  
  if is_camera_open():
    set_camera_open(False)
    #global move_sceneitem
    #obs.obs_source_release(move_sceneitem)
    
  else:
    set_camera_open(True)
    set_cap(cv2.VideoCapture(3))
    
def is_camera_open():
  return open_camera

def get_cap():
  return cap

def set_cap(cap_out):
  global cap
  
  cv2_version_major = int(cv2.__version__.split('.')[0])
  set_hands(mp_hands.Hands(model_complexity=1,max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5))

  if cv2_version_major > 3 :
    cap_out.set(cv2.CAP_PROP_FRAME_WIDTH,height )
    cap_out.set(cv2.CAP_PROP_FRAME_HEIGHT,width )
  else :  # before 3.0
    cap_out.set( cv2.cv.CV_CAP_PROP_FRAME_WIDTH, height)
    cap_out.set( cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, width)
  cap = cap_out
  
def get_hands():
  return hands

def set_hands(mp_hands):
  global hands
  hands = mp_hands
  
def calc_pos(results):
  
  multiply_coord = [1920,1080, -1]
  coords = np.multiply(
      np.array(
        (float("{:.4f}".format(results[0])), 
        float("{:.4f}".format(results[1])),
        float("{:.6f}".format(results[2]))
        ))
    , [multiply_coord[0],multiply_coord[1],multiply_coord[2]])
  moving_source(coords)

  return tuple(coords).astype(int)
    
def moving_source(coords):  

  global move_pos_trheshold, move_sceneitem, move_sceneitem_scale

  local_pos = obs.vec2()
  local_scale = obs.vec2()
    
  local_scale.x = (1+move_sceneitem_scale.x)* smooth_differ(coords[2], move_scale_threshold, "scale_x")*5
  local_scale.y = (1+move_sceneitem_scale.y) * smooth_differ(coords[2], move_scale_threshold, "scale_y")*5
  local_pos.x =   smooth_differ(coords[0], move_pos_trheshold, "pos_x")
  local_pos.y =   smooth_differ(coords[1], move_pos_trheshold, "pos_y")
  obs.obs_sceneitem_set_pos(move_sceneitem, local_pos)
  obs.obs_sceneitem_set_scale(move_sceneitem, local_scale)
      
def smooth_differ(value, percentage_cutoff, smooth_data_key):
    
  global smooth_data  
  aux_coord = [smooth_data.get(smooth_data_key), value]
  
  values_series = pd.Series(aux_coord)    
  difference = abs((values_series.pct_change()[1])*100)
  
  if difference < percentage_cutoff:

    return smooth_data[smooth_data_key]

  smooth_data[smooth_data_key] = value
      
  return value
  
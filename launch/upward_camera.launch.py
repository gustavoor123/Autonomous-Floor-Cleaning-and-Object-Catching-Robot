from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    video_device = LaunchConfiguration('video_device')
    publish_debug = LaunchConfiguration('publish_debug_image')

    # Arducam camera node
    camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='upward_camera',
        output='screen',
        parameters=[{
            'video_device': video_device,
            'image_size': [1280, 720],
            'pixel_format': 'YUYV',
            'camera_info_url': '',
        }],
        remappings=[
            ('/image_raw', '/upward_camera/image_raw'),
            ('/camera_info', '/upward_camera/camera_info'),
        ],
    )

    # trash_detector for ojects
    trash_detector = Node(
        package='my_bot',
        executable='trash_detector.py',
        name='trash_detector',
        output='screen',
        parameters=[{
            'target_class_id': 39,         
            'confidence_threshold': 0.25,  
            'publish_debug_image': publish_debug,
            'model_name': 'yolov8n.pt',
            'image_size': 416,             
            'queue_depth': 1,             
            'frame_skip': 1,              
            'detection_holdover': 0.4,     
            'yolo_reconfirm_frames': 30,   
            'use_tracker': True,          
            'detection_mode': 'yolo_only',  
            'motion_min_area_frac': 0.004,  
            'motion_max_area_frac': 0.4,    
            'motion_history': 300,         
            'motion_var_threshold': 45.0,  
            'motion_score': 0.40,           
            'motion_min_aspect': 0.0,       
            'motion_max_aspect': 100.0,    
            'motion_min_score': 0.0,        
            'motion_global_reject_frac': 0.20,  
            'motion_edge_margin_frac': 0.05,    
            'motion_central_bias': True,       
            'motion_huge_wide_area_frac': 1.0, 
            'motion_huge_wide_aspect': 0.0,  
            'motion_tall_bonus': False,         
            'motion_use_frame_diff': True,      
            'motion_diff_threshold': 18,      
            'motion_morph_kernel': 3,         
            'motion_downscale': 2,            
            # Tracker validation (anti-drift)
            'tracker_max_jump_frac': 0.20,
            'tracker_area_min_ratio': 0.4,
            'tracker_area_max_ratio': 2.5,
            'tracker_edge_margin': 4,
            'tracker_yolo_grace_sec': 1.5,
        }],
    )

    # trash_approach when object is detected
    trash_approach = Node(
        package='my_bot',
        executable='trash_approach.py',
        name='trash_approach',
        output='screen',
        parameters=[{
            'centre_tolerance': 0.05,
            'kp_angular': 12.0,   
            'kp_linear': 8.0,    
            'max_linear': 1.5,
            'max_angular': 3.5,
            'min_linear': 0.40, 
            'min_angular': 0.55,  
            'invert_y': False,
            'lost_timeout': 0.40, 
            'cmd_smoothing': 0.0, 
            'target_image_x': 0.50,
            'target_image_y': 0.64,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video0',
            description='V4L2 device path for the USB camera'),
        DeclareLaunchArgument(
            'publish_debug_image',
            default_value='false',
            description='Publish annotated debug image with bounding boxes'),
        camera_node,
        trash_detector,
        trash_approach,
    ])

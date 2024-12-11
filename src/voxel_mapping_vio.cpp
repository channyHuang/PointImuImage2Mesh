#include <iostream>

#include "voxel_mapping.hpp"

double g_vio_frame_cost_time = 0;
std::list<double> frame_cost_time_vec;
extern Global_map       g_map_rgb_pts_mesh;
std::deque<std::shared_ptr<Image_frame>> m_queue_image_with_pose;
StatesGroup g_lio_state;
std::shared_ptr< Common_tools::ThreadPool > m_thread_pool_ptr;

Eigen::Matrix<double, 3, 3, Eigen::RowMajor> m_camera_ext_R;
Eigen::Matrix<double, 3, 1> m_camera_ext_t;
#define PI_M (3.14159265358)

void Voxel_mapping::image_cbk(const sensor_msgs::ImageConstPtr &msg) {
    double timestamp = msg->header.stamp.toSec();
    m_last_timestamp_img = timestamp;

    cv::Mat temp_img = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 )->image.clone();
    if (!temp_img.empty()) {
        cv::Mat img = temp_img.clone();
        process_image( img, timestamp);
    } else {
        std::cout << "image_cbk image empty" << std::endl;
    }
    temp_img.release();
}

void Voxel_mapping::image_comp_cbk( const sensor_msgs::CompressedImageConstPtr &msg ) {
    double timestamp = msg->header.stamp.toSec();
    m_last_timestamp_img = timestamp;

    cv::Mat temp_img = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 )->image;
    if (!temp_img.empty()) {
        cv::Mat img = temp_img.clone();
        process_image( img, timestamp);
    } else {
        std::cout << "image_comp_cbk image empty" << std::endl;
    }
    temp_img.release();
}

void Voxel_mapping::process_image( cv::Mat &temp_img, double msg_time ) {
    std::shared_ptr< Image_frame > img_pose = std::make_shared< Image_frame >( g_cam_K );
    cv::remap( temp_img, img_pose->m_img, m_ud_map1, m_ud_map2, cv::INTER_LINEAR );
    img_pose->m_timestamp = msg_time;
    img_pose->init_cubic_interpolation();
    img_pose->image_equalize();
    m_camera_data_mutex.lock();
    m_queue_image_with_pose.push_back( img_pose );
    m_camera_data_mutex.unlock();
}

void Voxel_mapping::set_image_pose( std::shared_ptr< Image_frame > &image_pose, const StatesGroup &state_l2w ) {

    // current lidar to world
    mat_3_3 rot_mat = state_l2w.rot_end;
    vec_3   t_vec = state_l2w.pos_end;
    // world to current lidar 
    mat_3_3 w2l_R = rot_mat.inverse();
    vec_3 w2l_t = -rot_mat.inverse() * t_vec;
    // current camera to current lidar
    mat_3_3 rot = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( m_camera_ext_R.data() );
    vec_3 t = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( m_camera_ext_t.data() );
    // current lidar to current camera
    auto rot_ext_i2c = rot.inverse();
    auto pos_ext_i2c = rot.inverse() * (-t);

    vec_3   pose_t = rot_ext_i2c * (w2l_t) + pos_ext_i2c;
    mat_3_3 R_w2c = rot_ext_i2c * w2l_R;

    image_pose->set_pose( eigen_q( R_w2c ), pose_t );
    image_pose->fx = m_K[0];
    image_pose->fy = m_K[4];
    image_pose->cx = m_K[2];
    image_pose->cy = m_K[5];

    image_pose->m_cam_K << image_pose->fx, 0, image_pose->cx, 0, image_pose->fy, image_pose->cy, 0, 0, 1;
}

void Voxel_mapping::showImage(cv::Mat &mOriginImg, char* costTimeText) {
    cv::Mat m_showImg;
    cv::Size size = cv::Size(m_vio_image_width, m_vio_image_heigh);
    int maxWidth = 540;
    if (m_vio_image_width > maxWidth) {
        size = cv::Size(maxWidth, m_vio_image_heigh * maxWidth / m_vio_image_width);
    }
    cv::resize(mOriginImg, m_showImg, size);
    sprintf(costTimeText, "VIO Frame cost time : %lf ms\n", g_vio_frame_cost_time);
    cv::putText(m_showImg, costTimeText, cv::Size(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
    cv::imshow("Frame", m_showImg);
    cv::waitKey(1);
}

void Voxel_mapping::wait_render_thread_finish() {
    if ( m_render_thread != nullptr ) {
        m_render_thread->get(); // wait render thread to finish.
        // m_render_thread = nullptr;
    }
}

void Voxel_mapping::service_VIO_update() {
    m_thread_pool_ptr = std::make_shared<Common_tools::ThreadPool>(6, true, false); // At least 5 threads are needs, here we allocate 6 threads.

    double cam_k_scale = m_vio_scale_factor;
    g_cam_K << m_K[ 0 ] / cam_k_scale, m_K[ 1 ], m_K[ 2 ] / cam_k_scale, 
                m_K[ 3 ], m_K[ 4 ] / cam_k_scale, m_K[ 5 ] / cam_k_scale, 
                m_K[ 6 ], m_K[ 7 ], m_K[ 8 ];
    
    g_cam_dist = Eigen::Map< Eigen::Matrix< double, 5, 1 > >( m_dist.data() );

    cv::Mat m_intrinsic, m_dist_coeffs;
    cv::eigen2cv( g_cam_K, m_intrinsic );
    cv::eigen2cv( g_cam_dist, m_dist_coeffs );
    initUndistortRectifyMap( m_intrinsic, m_dist_coeffs, cv::Mat(), m_intrinsic, cv::Size( m_vio_image_width / m_vio_scale_factor, m_vio_image_heigh / m_vio_scale_factor ) , CV_16SC2, m_ud_map1, m_ud_map2 );

    // camera to lidar
    m_camera_ext_R = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( m_camera_extrin_R.data() ).inverse();
    m_camera_ext_t = - m_camera_ext_R * Eigen::Map< Eigen::Matrix< double, 3, 1 > >( m_camera_extrin_T.data() );

    
    g_map_rgb_pts_mesh.m_minimum_depth_for_projection = 0.1; // m_tracker_minimum_depth;
    g_map_rgb_pts_mesh.m_maximum_depth_for_projection = 200; // m_tracker_maximum_depth;

    Common_tools::Timer tim;
    cv::Mat             img_get;
    state.cam_intrinsic << g_cam_K( 0, 0 ), g_cam_K( 1, 1 ), g_cam_K( 0, 2 ), g_cam_K( 1, 2 );
    state.rot_ext_i2c = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >(m_camera_ext_R.data());
    state.pos_ext_i2c = Eigen::Map< Eigen::Matrix< double, 3, 1 > >(m_camera_ext_t.data());

    char costTimeText[100] = { 0 };

    ros::Subscriber sub_image = m_ros_node_ptr->subscribe( m_img_topic, 200000, &Voxel_mapping::image_cbk, this );
    // ros::Subscriber sub_image = m_ros_node_ptr->subscribe( m_img_topic, 200000, &Voxel_mapping::image_comp_cbk, this );

    while (1) {
        if ( m_queue_image_with_pose.size() == 0 ) {
            std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            std::this_thread::yield();
            continue;
        }

        m_camera_data_mutex.lock();
        std::shared_ptr< Image_frame > img_pose = m_queue_image_with_pose.front();
        m_queue_image_with_pose.pop_front();
        m_camera_data_mutex.unlock();

        img_pose->set_frame_idx( g_camera_frame_idx );
        
        
        if ( g_camera_frame_idx == 0 ) {
            std::vector< cv::Point2f >                pts_2d_vec;
            std::vector< std::shared_ptr< RGB_pts > > rgb_pts_vec;
            while ( ( ( g_map_rgb_pts_mesh.m_rgb_pts_vec.size() <= 100 ) ) ) {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            }
            set_image_pose( img_pose, state ); // For first frame pose, we suppose that the motion is static.
            g_camera_frame_idx++;
            continue;
        }
        
        g_camera_frame_idx++;
        showImage(img_pose->m_img, costTimeText);

        tim.tic( "Frame" );
        m_mutex_lio_process.lock();
        g_lio_state = state;
        m_mutex_lio_process.unlock();
        
        set_image_pose( img_pose, g_lio_state );
        
        wait_render_thread_finish();
       
        if ( 1 ) {
            tim.tic( "Render" );
            if ( 1 ) { // Using multiple threads for rendering
                g_map_rgb_pts_mesh.m_if_get_all_pts_in_boxes_using_mp = 0;
                // 把上一帧的点云
                m_render_thread = std::make_shared< std::shared_future< void > >( m_thread_pool_ptr->commit_task(
                    render_pts_in_voxels_mp, img_pose, &g_map_rgb_pts_mesh.m_voxels_recent_visited, img_pose->m_timestamp ) );
            } else {
                g_map_rgb_pts_mesh.m_if_get_all_pts_in_boxes_using_mp = 0;
            }
            g_map_rgb_pts_mesh.m_last_updated_frame_idx = img_pose->m_frame_idx;
        }
        g_map_rgb_pts_mesh.update_pose_for_projection( img_pose, -0.4 );
        double frame_cost = tim.toc( "Frame" );
        frame_cost_time_vec.push_back( frame_cost );

        double display_cost_time = std::accumulate( frame_cost_time_vec.begin(), frame_cost_time_vec.end(), 0.0 ) / frame_cost_time_vec.size();
        g_vio_frame_cost_time = display_cost_time;
    }
}

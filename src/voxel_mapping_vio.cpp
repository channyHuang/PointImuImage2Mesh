#include <iostream>

#include "voxel_mapping.hpp"

// image size
double m_vio_image_width = 1226;
double m_vio_image_heigh = 370;
// double m_vio_image_width = 1920;
// double m_vio_image_heigh = 1080;
double m_vio_scale_factor = 1.0;

// params
// CameraToLidar_TFMat_0
// std::vector< double > intrinsic_data = 
//     {1.019725357323183e+03,  0.0,        9.530492032695936e+02,
//     0.0,         1.020588984973907e+03, 5.432658976871853e+02,
//     0.0,         0.0,        1.0};
// std::vector< double > camera_dist_coeffs_data = {0.025681217868279, -0.018056404837187, 0.0, 0.0, 0.000000};
// // lidar (x,y,z) -> camera (z,-x,-y)
// std::vector< double > camera_ext_R_data = 
//     {-0.001046455132281, -0.999998679272604,  0.001243537173987,
//     0.226264330384285, -0.001448063248386, -0.974064862269746,
//     0.974065376516078, -7.379470683164596e-04, 0.226265546884003};
// std::vector< double > camera_ext_t_data =  {-0.063340671673354, -0.009431907279248, -0.108528849005723}; 

std::vector< double > intrinsic_data = 
    { m_vio_image_width,         0.0,         m_vio_image_width / 2.f,
        0.0,         m_vio_image_heigh,          m_vio_image_heigh / 2.f,
        0.0,         0.0,        1.0 };
std::vector< double > camera_dist_coeffs_data = {0.025681217868279, -0.018056404837187, 0.0, 0.0, 0.000000};
std::vector< double > camera_ext_R_data = 
    {0, -1, 0, 
     0, 0, -1,
     1, 0, 0};
std::vector< double > camera_ext_t_data =  {0, 0, 0};

// tracking
Rgbmap_tracker op_track;
int m_track_windows_size = 40;

// coordinate
Eigen::Matrix<double, 5, 1> g_cam_dist;
Eigen::Matrix<double, 3, 3, Eigen::RowMajor> m_camera_ext_R;
Eigen::Matrix<double, 3, 1> m_camera_ext_t;
Eigen::Matrix<double, 3, 3, Eigen::RowMajor> m_camera_to_lidar_R;
Eigen::Matrix<double, 3, 1> m_camera_to_lidar_t;
Eigen::Matrix3d rot_ext_i2c;
Eigen::Vector3d pos_ext_i2c;  

// cost time
double g_vio_frame_cost_time = 0;
std::list<double> frame_cost_time_vec;

extern Global_map       g_map_rgb_pts_mesh;
std::deque<std::shared_ptr<Image_frame>> m_queue_image_with_pose;
cv::Mat m_intrinsic, m_dist_coeffs;
cv::Mat m_ud_map1, m_ud_map2;
StatesGroup g_lio_state;
std::shared_ptr< Common_tools::ThreadPool > m_thread_pool_ptr;
int esikf_iter_times = 2;

#define PI_M (3.14159265358)

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

void set_image_pose( std::shared_ptr< Image_frame > &image_pose, const StatesGroup &state_l2w ) {

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
    rot_ext_i2c = rot.inverse();
    pos_ext_i2c = rot.inverse() * (-t);

    vec_3   pose_t = rot_ext_i2c * (w2l_t) + pos_ext_i2c;
    mat_3_3 R_w2c = rot_ext_i2c * w2l_R;

    image_pose->set_pose( eigen_q( R_w2c ), pose_t );
    image_pose->fx = intrinsic_data[0];
    image_pose->fy = intrinsic_data[4];
    image_pose->cx = intrinsic_data[2];
    image_pose->cy = intrinsic_data[5];

    image_pose->m_cam_K << image_pose->fx, 0, image_pose->cx, 0, image_pose->fy, image_pose->cy, 0, 0, 1;

    // FILE *fp = fopen(("/home/" + std::to_string(image_pose->m_frame_idx) + "state.txt").c_str(), "w+");
    // if (fp)
    // {
    //     auto q = eigen_q(state_l2w.rot_end);
    //     fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf\r\n", q.w(), q.x(), q.y(), q.z(), 
    //     state_l2w.pos_end(0), state_l2w.pos_end(1), state_l2w.pos_end(2));
    //     fclose(fp);
    // }
}

void showImage(cv::Mat &mOriginImg, char* costTimeText) {
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

bool Voxel_mapping::vio_preintegration( StatesGroup &state_in, StatesGroup &state_out, double current_frame_time ) {
    state_out = state_in;
    // if ( current_frame_time <= state_in.last_update_time ) {
    //     // cout << ANSI_COLOR_RED_BOLD << "Error current_frame_time <= state_in.last_update_time | " <<
    //     // current_frame_time - state_in.last_update_time << ANSI_COLOR_RESET << endl;
    //     return false;
    // }
    m_mutex_buffer.lock();
    std::deque< sensor_msgs::Imu::ConstPtr > vio_imu_queue;
    for ( auto it = imu_buffer_vio.begin(); it != imu_buffer_vio.end(); it++ ) {
        vio_imu_queue.push_back( *it );
        if ( ( *it )->header.stamp.toSec() > current_frame_time ) {
            break;
        }
    }

    while ( !imu_buffer_vio.empty() ) {
        double imu_time = imu_buffer_vio.front()->header.stamp.toSec();
        if ( imu_time < current_frame_time - 0.2 ) {
            imu_buffer_vio.pop_front();
        } else {
            break;
        }
    }
    // cout << "Current VIO_imu buffer size = " << imu_buffer_vio.size() << endl;
    state_out = m_imu_process->imu_preintegration( state_out, vio_imu_queue, current_frame_time - vio_imu_queue.back()->header.stamp.toSec() );
    eigen_q q_diff( state_out.rot_end.transpose() * state_in.rot_end );
    // cout << "Pos diff = " << (state_out.pos_end - state_in.pos_end).transpose() << endl;
    // cout << "Euler diff = " << q_diff.angularDistance(eigen_q::Identity()) * 57.3 << endl;
    m_mutex_buffer.unlock();
    // state_out.last_update_time = current_frame_time;
    return true;
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
    g_cam_K << intrinsic_data[ 0 ] / cam_k_scale, intrinsic_data[ 1 ], intrinsic_data[ 2 ] / cam_k_scale, 
                intrinsic_data[ 3 ], intrinsic_data[ 4 ] / cam_k_scale, intrinsic_data[ 5 ] / cam_k_scale, 
                intrinsic_data[ 6 ], intrinsic_data[ 7 ], intrinsic_data[ 8 ];
    
    g_cam_dist = Eigen::Map< Eigen::Matrix< double, 5, 1 > >( camera_dist_coeffs_data.data() );
    // lidar to camera
    m_camera_to_lidar_R = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( camera_ext_R_data.data() );
    m_camera_to_lidar_t = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( camera_ext_t_data.data() );
    // camera to lidar
    m_camera_ext_R = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( camera_ext_R_data.data() ).inverse();
    m_camera_ext_t = - m_camera_ext_R * Eigen::Map< Eigen::Matrix< double, 3, 1 > >( camera_ext_t_data.data() );

    cv::eigen2cv( g_cam_K, m_intrinsic );
    cv::eigen2cv( g_cam_dist, m_dist_coeffs );
    initUndistortRectifyMap( m_intrinsic, m_dist_coeffs, cv::Mat(), m_intrinsic, cv::Size( m_vio_image_width / m_vio_scale_factor, m_vio_image_heigh / m_vio_scale_factor ) , CV_16SC2, m_ud_map1, m_ud_map2 );
    // Init cv windows for debug
    op_track.set_intrinsic( g_cam_K, g_cam_dist * 0, cv::Size( m_vio_image_width / m_vio_scale_factor, m_vio_image_heigh / m_vio_scale_factor ) );
    // op_track.m_maximum_vio_tracked_pts = m_maximum_vio_tracked_pts;
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
        // while ( m_queue_image_with_pose.size() > m_maximum_image_buffer ) {
        //     cout << ANSI_COLOR_BLUE_BOLD << "=== Pop image! current queue size = " << m_queue_image_with_pose.size() << " ===" << ANSI_COLOR_RESET
        //          << endl;
        //     op_track.track_img( m_queue_image_with_pose.front(), -20 );
        //     m_queue_image_with_pose.pop_front();
        // }
        std::shared_ptr< Image_frame > img_pose = m_queue_image_with_pose.front();
        // double                             message_time = img_pose->m_timestamp;
        m_queue_image_with_pose.pop_front();
        m_camera_data_mutex.unlock();

        // g_camera_lidar_queue.m_last_visual_time = img_pose->m_timestamp + g_lio_state.td_ext_i2c;

        img_pose->set_frame_idx( g_camera_frame_idx );
        
        
        if ( g_camera_frame_idx == 0 ) {
            std::vector< cv::Point2f >                pts_2d_vec;
            std::vector< std::shared_ptr< RGB_pts > > rgb_pts_vec;
            // while ( ( m_map_rgb_pts.is_busy() ) || ( ( m_map_rgb_pts.m_rgb_pts_vec.size() <= 100 ) ) )
            while ( ( ( g_map_rgb_pts_mesh.m_rgb_pts_vec.size() <= 100 ) ) ) {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            }
            set_image_pose( img_pose, state ); // For first frame pose, we suppose that the motion is static.
            g_map_rgb_pts_mesh.selection_points_for_projection( img_pose, &rgb_pts_vec, &pts_2d_vec, m_track_windows_size / m_vio_scale_factor );
            // op_track.init( img_pose, rgb_pts_vec, pts_2d_vec );

            // if (rgb_pts_vec.size() < 10) {
            //     continue;
            // }
            g_camera_frame_idx++;
            continue;
        }
        
        g_camera_frame_idx++;
        showImage(img_pose->m_img, costTimeText);
        tim.tic( "Wait" );

        // while ( g_camera_lidar_queue.if_camera_can_process() == false ) {
        //     std::this_thread::sleep_for( std::chrono::milliseconds( THREAD_SLEEP_TIM ) );
        //     std::this_thread::yield();
        //     cv_keyboard_callback();
        // }

        tim.tic( "Frame" );
        tim.tic( "Track_img" );
        m_mutex_lio_process.lock();
        g_lio_state = state;
        m_mutex_lio_process.unlock();
        // 积分计算摄像机位姿
        StatesGroup state_out = g_lio_state;
        
        // m_cam_measurement_weight = std::max( 0.001, std::min( 5.0 / m_number_of_new_visited_voxel, 0.01 ) );
        // if ( vio_preintegration( g_lio_state, state_out, img_pose->m_timestamp + g_lio_state.td_ext_i2c ) == false ) {
            // m_mutex_lio_process.unlock();
            //continue;
        // }
        set_image_pose( img_pose, state_out );
        
        // op_track.track_img( img_pose, -20 );
        tim.tic( "Ransac" );
        set_image_pose( img_pose, state_out );
        
        // ANCHOR -  remove point using PnP.
        // if ( op_track.remove_outlier_using_ransac_pnp( img_pose ) == 0 ) {
            // cout << ANSI_COLOR_RED_BOLD << g_camera_frame_idx << "****** Remove_outlier_using_ransac_pnp error*****" << ANSI_COLOR_RESET << endl;
        // } 
        tim.tic( "Vio_f2f" );
        bool res_esikf = true, res_photometric = true;
        
        wait_render_thread_finish();
        res_esikf = vio_esikf( state_out, op_track );
        tim.tic( "Vio_f2m" );
        res_photometric = vio_photometric( state_out, op_track, img_pose );
        g_lio_state = state_out;
        // print_dash_board();
        set_image_pose( img_pose, state_out );
        
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

            tim.tic( "Mvs_record" );
            // if ( m_if_record_mvs ) {
            //     g_mutex_render.lock();
                
            //     m_mvs_recorder.insert_image_and_pts( img_pose, g_map_rgb_pts_mesh.m_pts_last_hitted );

            //     g_mutex_render.unlock();
            // } 
        }
        // ANCHOR - render point cloud
        // m_mutex_lio_process.unlock();
        g_map_rgb_pts_mesh.update_pose_for_projection( img_pose, -0.4 );
        op_track.update_and_append_track_pts( img_pose, g_map_rgb_pts_mesh, m_track_windows_size / m_vio_scale_factor, 1000000 );
        double frame_cost = tim.toc( "Frame" );
        frame_cost_time_vec.push_back( frame_cost );

        double display_cost_time = std::accumulate( frame_cost_time_vec.begin(), frame_cost_time_vec.end(), 0.0 ) / frame_cost_time_vec.size();
        g_vio_frame_cost_time = display_cost_time;
        
    }
}

// ANCHOR - huber_loss
double get_huber_loss_scale( double reprojection_error, double outlier_threshold = 1.0 )
{
    // http://ceres-solver.org/nnls_modeling.html#lossfunction
    double scale = 1.0;
    if ( reprojection_error / outlier_threshold < 1.0 )
    {
        scale = 1.0;
    }
    else
    {
        scale = ( 2 * sqrt( reprojection_error ) / sqrt( outlier_threshold ) - 1.0 ) / reprojection_error;
    }
    return scale;
}
    
// ANCHOR - VIO_esikf
const int minimum_iteration_pts = 10;
bool      Voxel_mapping::vio_esikf( StatesGroup &state_in, Rgbmap_tracker &op_track )
{
    Common_tools::Timer tim;
    tim.tic();
    scope_color( ANSI_COLOR_BLUE_BOLD );
    StatesGroup state_iter = state_in;
    // if ( !m_if_estimate_intrinsic ) // When disable the online intrinsic calibration.
    // {
        state_iter.cam_intrinsic << g_cam_K( 0, 0 ), g_cam_K( 1, 1 ), g_cam_K( 0, 2 ), g_cam_K( 1, 2 );
    // }

    // if ( !m_if_estimate_i2c_extrinsic )
    // {
    //     state_iter.pos_ext_i2c = m_inital_pos_ext_i2c;
    //     state_iter.rot_ext_i2c = m_inital_rot_ext_i2c;
    // }
    state_iter.pos_ext_i2c = pos_ext_i2c;
    state_iter.rot_ext_i2c = rot_ext_i2c;

    Eigen::Matrix< double, -1, -1 >                       H_mat;
    Eigen::Matrix< double, -1, 1 >                        meas_vec;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;
    Eigen::Matrix< double, DIM_OF_STATES, 1 >             solution;
    Eigen::Matrix< double, -1, -1 >                       K, KH;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > K_1;

    Eigen::SparseMatrix< double > H_mat_spa, H_T_H_spa, K_spa, KH_spa, vec_spa, I_STATE_spa;
    I_STATE.setIdentity();
    I_STATE_spa = I_STATE.sparseView();
    double fx, fy, cx, cy, time_td;

    int                   total_pt_size = op_track.m_map_rgb_pts_in_current_frame_pos.size();
    std::vector< double > last_reprojection_error_vec( total_pt_size ), current_reprojection_error_vec( total_pt_size );

    if ( total_pt_size < minimum_iteration_pts )
    {
        state_in = state_iter;
        return false;
    }
    H_mat.resize( total_pt_size * 2, DIM_OF_STATES );
    meas_vec.resize( total_pt_size * 2, 1 );
    double last_repro_err = 3e8;
    int    avail_pt_count = 0;
    double last_avr_repro_err = 0;

    double acc_reprojection_error = 0;
    double img_res_scale = 1.0;
    for ( int iter_count = 0; iter_count < esikf_iter_times; iter_count++ )
    {

        // cout << "========== Iter " << iter_count << " =========" << endl;
        mat_3_3 R_imu = state_iter.rot_end;
        vec_3   t_imu = state_iter.pos_end;
        vec_3   t_c2w = R_imu * state_iter.pos_ext_i2c + t_imu;
        mat_3_3 R_c2w = R_imu * state_iter.rot_ext_i2c; // world to camera frame

        fx = state_iter.cam_intrinsic( 0 );
        fy = state_iter.cam_intrinsic( 1 );
        cx = state_iter.cam_intrinsic( 2 );
        cy = state_iter.cam_intrinsic( 3 );
        time_td = state_iter.td_ext_i2c_delta;

        vec_3   t_w2c = -R_c2w.transpose() * t_c2w;
        mat_3_3 R_w2c = R_c2w.transpose();
        int     pt_idx = -1;
        acc_reprojection_error = 0;
        vec_3               pt_3d_w, pt_3d_cam;
        vec_2               pt_img_measure, pt_img_proj, pt_img_vel;
        eigen_mat_d< 2, 3 > mat_pre;
        eigen_mat_d< 3, 3 > mat_A, mat_B, mat_C, mat_D, pt_hat;
        H_mat.setZero();
        solution.setZero();
        meas_vec.setZero();
        avail_pt_count = 0;
        for ( auto it = op_track.m_map_rgb_pts_in_last_frame_pos.begin(); it != op_track.m_map_rgb_pts_in_last_frame_pos.end(); it++ )
        {
            pt_3d_w = ( ( RGB_pts * ) it->first )->get_pos();
            pt_img_vel = ( ( RGB_pts * ) it->first )->m_img_vel;
            pt_img_measure = vec_2( it->second.x, it->second.y );
            pt_3d_cam = R_w2c * pt_3d_w + t_w2c;
            pt_img_proj = vec_2( fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ) + cx, fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 ) + cy ) + time_td * pt_img_vel;
            double repro_err = ( pt_img_proj - pt_img_measure ).norm();
            double huber_loss_scale = get_huber_loss_scale( repro_err );
            pt_idx++;
            acc_reprojection_error += repro_err;
            // if (iter_count == 0 || ((repro_err - last_reprojection_error_vec[pt_idx]) < 1.5))
            if ( iter_count == 0 || ( ( repro_err - last_avr_repro_err * 5.0 ) < 0 ) )
            {
                last_reprojection_error_vec[ pt_idx ] = repro_err;
            }
            else
            {
                last_reprojection_error_vec[ pt_idx ] = repro_err;
            }
            avail_pt_count++;
            // Appendix E of r2live_Supplementary_material.
            // https://github.com/hku-mars/r2live/blob/master/supply/r2live_Supplementary_material.pdf
            mat_pre << fx / pt_3d_cam( 2 ), 0, -fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ), 0, fy / pt_3d_cam( 2 ), -fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 );

            pt_hat = Sophus::SO3d::hat( ( R_imu.transpose() * ( pt_3d_w - t_imu ) ) );
            mat_A = state_iter.rot_ext_i2c.transpose() * pt_hat;
            mat_B = -state_iter.rot_ext_i2c.transpose() * ( R_imu.transpose() );
            mat_C = Sophus::SO3d::hat( pt_3d_cam );
            mat_D = -state_iter.rot_ext_i2c.transpose();
            meas_vec.block( pt_idx * 2, 0, 2, 1 ) = ( pt_img_proj - pt_img_measure ) * huber_loss_scale / img_res_scale;

            H_mat.block( pt_idx * 2, 0, 2, 3 ) = mat_pre * mat_A * huber_loss_scale;
            H_mat.block( pt_idx * 2, 3, 2, 3 ) = mat_pre * mat_B * huber_loss_scale;
            if ( DIM_OF_STATES > 24 )
            {
                // Estimate time td.
                H_mat.block( pt_idx * 2, 24, 2, 1 ) = pt_img_vel * huber_loss_scale;
                // H_mat(pt_idx * 2, 24) = pt_img_vel(0) * huber_loss_scale;
                // H_mat(pt_idx * 2 + 1, 24) = pt_img_vel(1) * huber_loss_scale;
            }
            // if ( m_if_estimate_i2c_extrinsic )
            // {
            //     H_mat.block( pt_idx * 2, 18, 2, 3 ) = mat_pre * mat_C * huber_loss_scale;
            //     H_mat.block( pt_idx * 2, 21, 2, 3 ) = mat_pre * mat_D * huber_loss_scale;
            // }

            // if ( m_if_estimate_intrinsic )
            // {
            //     H_mat( pt_idx * 2, 25 ) = pt_3d_cam( 0 ) / pt_3d_cam( 2 ) * huber_loss_scale;
            //     H_mat( pt_idx * 2 + 1, 26 ) = pt_3d_cam( 1 ) / pt_3d_cam( 2 ) * huber_loss_scale;
            //     H_mat( pt_idx * 2, 27 ) = 1 * huber_loss_scale;
            //     H_mat( pt_idx * 2 + 1, 28 ) = 1 * huber_loss_scale;
            // }
        }
        H_mat = H_mat / img_res_scale;
        acc_reprojection_error /= total_pt_size;

        last_avr_repro_err = acc_reprojection_error;
        if ( avail_pt_count < minimum_iteration_pts )
        {
            break;
        }

        H_mat_spa = H_mat.sparseView();
        Eigen::SparseMatrix< double > Hsub_T_temp_mat = H_mat_spa.transpose();
        vec_spa = ( state_iter - state_in ).sparseView();
        H_T_H_spa = Hsub_T_temp_mat * H_mat_spa;
        // Notice that we have combine some matrix using () in order to boost the matrix multiplication.
        Eigen::SparseMatrix< double > temp_inv_mat =
            ( ( H_T_H_spa.toDense() + eigen_mat< -1, -1 >( state_in.cov * 0.01 ).inverse() ).inverse() ).sparseView();
        KH_spa = temp_inv_mat * ( Hsub_T_temp_mat * H_mat_spa );
        solution = ( temp_inv_mat * ( Hsub_T_temp_mat * ( ( -1 * meas_vec.sparseView() ) ) ) - ( I_STATE_spa - KH_spa ) * vec_spa ).toDense();

        state_iter = state_iter + solution;

        if ( fabs( acc_reprojection_error - last_repro_err ) < 0.01 )
        {
            break;
        }
        last_repro_err = acc_reprojection_error;
    }

    if ( avail_pt_count >= minimum_iteration_pts )
    {
        state_iter.cov = ( ( I_STATE_spa - KH_spa ) * state_iter.cov.sparseView() ).toDense();
    }

    state_iter.td_ext_i2c += state_iter.td_ext_i2c_delta;
    state_iter.td_ext_i2c_delta = 0;
    state_in = state_iter;
    return true;
}


bool Voxel_mapping::vio_photometric( StatesGroup &state_in, Rgbmap_tracker &op_track, std::shared_ptr< Image_frame > &image )
{
    bool m_if_estimate_intrinsic = false;
    Common_tools::Timer tim;
    tim.tic();
    StatesGroup state_iter = state_in;
    if ( !m_if_estimate_intrinsic )     // When disable the online intrinsic calibration.
    {
        state_iter.cam_intrinsic << g_cam_K( 0, 0 ), g_cam_K( 1, 1 ), g_cam_K( 0, 2 ), g_cam_K( 1, 2 );
    }
    // if ( !m_if_estimate_i2c_extrinsic ) // When disable the online extrinsic calibration.
    // {
    //     state_iter.pos_ext_i2c = m_inital_pos_ext_i2c;
    //     state_iter.rot_ext_i2c = m_inital_rot_ext_i2c;
    // }
    state_iter.pos_ext_i2c = pos_ext_i2c;
    state_iter.rot_ext_i2c = rot_ext_i2c;


    Eigen::Matrix< double, -1, -1 >                       H_mat, R_mat_inv;
    Eigen::Matrix< double, -1, 1 >                        meas_vec;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;
    Eigen::Matrix< double, DIM_OF_STATES, 1 >             solution;
    Eigen::Matrix< double, -1, -1 >                       K, KH;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > K_1;
    Eigen::SparseMatrix< double >                         H_mat_spa, H_T_H_spa, R_mat_inv_spa, K_spa, KH_spa, vec_spa, I_STATE_spa;
    I_STATE.setIdentity();
    I_STATE_spa = I_STATE.sparseView();
    double fx, fy, cx, cy, time_td;

    int                   total_pt_size = op_track.m_map_rgb_pts_in_current_frame_pos.size();
    std::vector< double > last_reprojection_error_vec( total_pt_size ), current_reprojection_error_vec( total_pt_size );
    if ( total_pt_size < minimum_iteration_pts )
    {
        state_in = state_iter;
        return false;
    }

    int err_size = 3;
    H_mat.resize( total_pt_size * err_size, DIM_OF_STATES );
    meas_vec.resize( total_pt_size * err_size, 1 );
    R_mat_inv.resize( total_pt_size * err_size, total_pt_size * err_size );

    double last_repro_err = 3e8;
    int    avail_pt_count = 0;
    double last_avr_repro_err = 0;
    int    if_esikf = 1;

    double acc_photometric_error = 0;
#if DEBUG_PHOTOMETRIC
    printf("==== [Image frame %d] ====\r\n", g_camera_frame_idx);
#endif
    for ( int iter_count = 0; iter_count < 2; iter_count++ )
    {
        mat_3_3 R_imu = state_iter.rot_end;
        vec_3   t_imu = state_iter.pos_end;
        vec_3   t_c2w = R_imu * state_iter.pos_ext_i2c + t_imu;
        mat_3_3 R_c2w = R_imu * state_iter.rot_ext_i2c; // world to camera frame

        fx = state_iter.cam_intrinsic( 0 );
        fy = state_iter.cam_intrinsic( 1 );
        cx = state_iter.cam_intrinsic( 2 );
        cy = state_iter.cam_intrinsic( 3 );
        time_td = state_iter.td_ext_i2c_delta;

        vec_3   t_w2c = -R_c2w.transpose() * t_c2w;
        mat_3_3 R_w2c = R_c2w.transpose();
        int     pt_idx = -1;
        acc_photometric_error = 0;
        vec_3               pt_3d_w, pt_3d_cam;
        vec_2               pt_img_measure, pt_img_proj, pt_img_vel;
        eigen_mat_d< 2, 3 > mat_pre;
        eigen_mat_d< 3, 2 > mat_photometric;
        eigen_mat_d< 3, 3 > mat_d_pho_d_img;
        eigen_mat_d< 3, 3 > mat_A, mat_B, mat_C, mat_D, pt_hat;
        R_mat_inv.setZero();
        H_mat.setZero();
        solution.setZero();
        meas_vec.setZero();
        avail_pt_count = 0;
        int iter_layer = 0;
        tim.tic( "Build_cost" );
        for ( auto it = op_track.m_map_rgb_pts_in_last_frame_pos.begin(); it != op_track.m_map_rgb_pts_in_last_frame_pos.end(); it++ )
        {
            if ( ( ( RGB_pts * ) it->first )->m_N_rgb < 3 )
            {
                continue;
            }
            pt_idx++;
            pt_3d_w = ( ( RGB_pts * ) it->first )->get_pos();
            pt_img_vel = ( ( RGB_pts * ) it->first )->m_img_vel;
            pt_img_measure = vec_2( it->second.x, it->second.y );
            pt_3d_cam = R_w2c * pt_3d_w + t_w2c;
            pt_img_proj = vec_2( fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ) + cx, fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 ) + cy ) + time_td * pt_img_vel;
            if (pt_img_proj(0) >= m_vio_image_width - 6 || pt_img_proj(1) >= m_vio_image_heigh - 6 || pt_img_proj(0) <= 5 || pt_img_proj(1) <= 5) {
                continue;
            }

            vec_3   pt_rgb = ( ( RGB_pts * ) it->first )->get_rgb();
            mat_3_3 pt_rgb_info = mat_3_3::Zero();
            mat_3_3 pt_rgb_cov = ( ( RGB_pts * ) it->first )->get_rgb_cov();
            for ( int i = 0; i < 3; i++ )
            {
                pt_rgb_info( i, i ) = 1.0 / pt_rgb_cov( i, i ) ;
                R_mat_inv( pt_idx * err_size + i, pt_idx * err_size + i ) = pt_rgb_info( i, i );
                // R_mat_inv( pt_idx * err_size + i, pt_idx * err_size + i ) =  1.0;
            }
            vec_3  obs_rgb_dx, obs_rgb_dy;
            vec_3  obs_rgb = image->get_rgb( pt_img_proj( 0 ), pt_img_proj( 1 ), 0, &obs_rgb_dx, &obs_rgb_dy );
            vec_3  photometric_err_vec = ( obs_rgb - pt_rgb );
            double huber_loss_scale = get_huber_loss_scale( photometric_err_vec.norm() );
            photometric_err_vec *= huber_loss_scale;
            double photometric_err = photometric_err_vec.transpose() * pt_rgb_info * photometric_err_vec;

            acc_photometric_error += photometric_err;

            last_reprojection_error_vec[ pt_idx ] = photometric_err;

            mat_photometric.setZero();
            mat_photometric.col( 0 ) = obs_rgb_dx;
            mat_photometric.col( 1 ) = obs_rgb_dy;

            avail_pt_count++;
            mat_pre << fx / pt_3d_cam( 2 ), 0, -fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ), 0, fy / pt_3d_cam( 2 ), -fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 );
            mat_d_pho_d_img = mat_photometric * mat_pre;

            pt_hat = Sophus::SO3d::hat( ( R_imu.transpose() * ( pt_3d_w - t_imu ) ) );
            mat_A = state_iter.rot_ext_i2c.transpose() * pt_hat;
            mat_B = -state_iter.rot_ext_i2c.transpose() * ( R_imu.transpose() );
            mat_C = Sophus::SO3d::hat( pt_3d_cam );
            mat_D = -state_iter.rot_ext_i2c.transpose();
            meas_vec.block( pt_idx * 3, 0, 3, 1 ) = photometric_err_vec ;

            H_mat.block( pt_idx * 3, 0, 3, 3 ) = mat_d_pho_d_img * mat_A * huber_loss_scale;
            H_mat.block( pt_idx * 3, 3, 3, 3 ) = mat_d_pho_d_img * mat_B * huber_loss_scale;
            if ( 1 )
            {
                // if ( m_if_estimate_i2c_extrinsic )
                // {
                //     H_mat.block( pt_idx * 3, 18, 3, 3 ) = mat_d_pho_d_img * mat_C * huber_loss_scale;
                //     H_mat.block( pt_idx * 3, 21, 3, 3 ) = mat_d_pho_d_img * mat_D * huber_loss_scale;
                // }
            }
        }
        R_mat_inv_spa = R_mat_inv.sparseView();
       
        last_avr_repro_err = acc_photometric_error;
        if ( avail_pt_count < minimum_iteration_pts )
        {
            break;
        }
        // Esikf
        tim.tic( "Iter" );
        if ( if_esikf )
        {
            H_mat_spa = H_mat.sparseView();
            Eigen::SparseMatrix< double > Hsub_T_temp_mat = H_mat_spa.transpose();
            vec_spa = ( state_iter - state_in ).sparseView();
            H_T_H_spa = Hsub_T_temp_mat * R_mat_inv_spa * H_mat_spa;
            Eigen::SparseMatrix< double > temp_inv_mat =
                ( H_T_H_spa.toDense() + ( state_in.cov * 0.001 ).inverse() ).inverse().sparseView();
            // ( H_T_H_spa.toDense() + ( state_in.cov ).inverse() ).inverse().sparseView();
            Eigen::SparseMatrix< double > Ht_R_inv = ( Hsub_T_temp_mat * R_mat_inv_spa );
            KH_spa = temp_inv_mat * Ht_R_inv * H_mat_spa;
            solution = ( temp_inv_mat * ( Ht_R_inv * ( ( -1 * meas_vec.sparseView() ) ) ) - ( I_STATE_spa - KH_spa ) * vec_spa ).toDense();
        }
        state_iter = state_iter + solution;
#if DEBUG_PHOTOMETRIC
        cout << "Average photometric error: " <<  acc_photometric_error / total_pt_size << endl;
        cout << "Solved solution: "<< solution.transpose() << endl;
#else
        if ( ( acc_photometric_error / total_pt_size ) < 10 ) // By experience.
        {
            break;
        }
#endif
        if ( fabs( acc_photometric_error - last_repro_err ) < 0.01 )
        {
            break;
        }
        last_repro_err = acc_photometric_error;
    }
    if ( if_esikf && avail_pt_count >= minimum_iteration_pts )
    {
        state_iter.cov = ( ( I_STATE_spa - KH_spa ) * state_iter.cov.sparseView() ).toDense();
    }
    state_iter.td_ext_i2c += state_iter.td_ext_i2c_delta;
    state_iter.td_ext_i2c_delta = 0;
    state_in = state_iter;
    return true;
}

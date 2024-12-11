#include "voxel_mapping.hpp"

void Voxel_mapping::pointBodyToWorld( const PointType &pi, PointType &po ) {
    V3D p_body( pi.x, pi.y, pi.z );

    V3D p_global( state.rot_end * ( m_extR * p_body + m_extT ) + state.pos_end );

    po.x = p_global( 0 );
    po.y = p_global( 1 );
    po.z = p_global( 2 );
    po.intensity = pi.intensity;
}

void Voxel_mapping::frameBodyToWorld( const PointCloudXYZI::Ptr &pi, PointCloudXYZI::Ptr &po ) {
    int pi_size = pi->points.size();
    po->resize( pi_size );
    for ( int i = 0; i < pi_size; i++ ) {
        /* transform to world frame */
        pointBodyToWorld( pi->points[ i ], po->points[ i ] );
    }
}

void Voxel_mapping::RGBpointBodyToWorld( PointType const *const pi, PointType *const po ) {
    V3D p_body( pi->x, pi->y, pi->z );
#ifdef USE_IKFOM
    // state_ikfom transfer_state = kf.get_x();
    V3D p_global( state_point.rot * ( state_point.offset_R_L_I * p_body + state_point.offset_T_L_I ) + state_point.pos );
#else
    V3D p_global( state.rot_end * ( m_extR * p_body + m_extT ) + state.pos_end );
#endif

    p_global = _gravity_correct_rotM * p_global;

    po->x = p_global( 0 );
    po->y = p_global( 1 );
    po->z = p_global( 2 );
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - floor( intensity );

    int reflection_map = intensity * 10000;
}

void Voxel_mapping::RGBpointBodyLidarToIMU( PointType const *const pi, PointType *const po ) {
    V3D p_body_lidar( pi->x, pi->y, pi->z );
#ifdef USE_IKFOM
    // state_ikfom transfer_state = kf.get_x();
    V3D p_body_imu( state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I );
#else
    V3D p_body_imu( m_extR * p_body_lidar + m_extT );
#endif

    po->x = p_body_imu( 0 );
    po->y = p_body_imu( 1 );
    po->z = p_body_imu( 2 );
    po->intensity = pi->intensity;
}

void Voxel_mapping::points_cache_collect() {
    PointVector points_history;
    m_ikdtree.acquire_removed_points( points_history );
    m_points_cache_size = points_history.size();
}

void Voxel_mapping::laser_map_fov_segment() {
    m_cub_need_rm.clear();
    m_kdtree_delete_counter = 0;
    m_kdtree_delete_time = 0.0;
    pointBodyToWorld( m_XAxis_Point_body, m_XAxis_Point_world );
#ifdef USE_IKFOM
    // state_ikfom fov_state = kf.get_x();
    // V3D pos_LiD = fov_state.pos + fov_state.rot * fov_state.offset_T_L_I;
    V3D pos_LiD = pos_lid;
#else
    V3D pos_LiD = state.pos_end;
#endif
    if ( !m_localmap_Initialized ) {
        for ( int i = 0; i < 3; i++ ) {
            m_LocalMap_Points.vertex_min[ i ] = pos_LiD( i ) - m_cube_len / 2.0;
            m_LocalMap_Points.vertex_max[ i ] = pos_LiD( i ) + m_cube_len / 2.0;
        }
        m_localmap_Initialized = true;
        return;
    }

    float dist_to_map_edge[ 3 ][ 2 ];
    bool  need_move = false;
    for ( int i = 0; i < 3; i++ ) {
        dist_to_map_edge[ i ][ 0 ] = fabs( pos_LiD( i ) - m_LocalMap_Points.vertex_min[ i ] );
        dist_to_map_edge[ i ][ 1 ] = fabs( pos_LiD( i ) - m_LocalMap_Points.vertex_max[ i ] );
        if ( dist_to_map_edge[ i ][ 0 ] <= MOV_THRESHOLD * DETECTION_RANGE || dist_to_map_edge[ i ][ 1 ] <= MOV_THRESHOLD * DETECTION_RANGE )
            need_move = true;
    }
    if ( !need_move )
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = m_LocalMap_Points;
    float mov_dist = max( ( m_cube_len - 2.0 * MOV_THRESHOLD * DETECTION_RANGE ) * 0.5 * 0.9, double( DETECTION_RANGE * ( MOV_THRESHOLD - 1 ) ) );
    for ( int i = 0; i < 3; i++ ) {
        tmp_boxpoints = m_LocalMap_Points;
        if ( dist_to_map_edge[ i ][ 0 ] <= MOV_THRESHOLD * DETECTION_RANGE ) {
            New_LocalMap_Points.vertex_max[ i ] -= mov_dist;
            New_LocalMap_Points.vertex_min[ i ] -= mov_dist;
            tmp_boxpoints.vertex_min[ i ] = m_LocalMap_Points.vertex_max[ i ] - mov_dist;
            m_cub_need_rm.push_back( tmp_boxpoints );
        } else if ( dist_to_map_edge[ i ][ 1 ] <= MOV_THRESHOLD * DETECTION_RANGE ) {
            New_LocalMap_Points.vertex_max[ i ] += mov_dist;
            New_LocalMap_Points.vertex_min[ i ] += mov_dist;
            tmp_boxpoints.vertex_max[ i ] = m_LocalMap_Points.vertex_min[ i ] + mov_dist;
            m_cub_need_rm.push_back( tmp_boxpoints );
        }
    }
    m_LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if ( m_cub_need_rm.size() > 0 )
        m_kdtree_delete_counter = m_ikdtree.Delete_Point_Boxes( m_cub_need_rm );
    m_kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void Voxel_mapping::standard_pcl_cbk( const sensor_msgs::PointCloud2::ConstPtr &msg ) {
    if ( !m_lidar_en ) {
        return;
    }

    m_mutex_buffer.lock();
    if ( msg->header.stamp.toSec() < m_last_timestamp_lidar ) {
        ROS_ERROR( "lidar loop back, clear buffer" );
        m_lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr( new PointCloudXYZI() );
    m_p_pre->process( msg, ptr );
    m_lidar_buffer.push_back( ptr );
    m_time_buffer.push_back( msg->header.stamp.toSec() );
    m_last_timestamp_lidar = msg->header.stamp.toSec();

    m_mutex_buffer.unlock();
    m_sig_buffer.notify_all();
}

void Voxel_mapping::imu_cbk( const sensor_msgs::Imu::ConstPtr &msg_in ) {
    if ( !m_imu_en )
        return;

    if ( m_last_timestamp_lidar < 0.0 )
        return;
    m_publish_count++;
    sensor_msgs::Imu::Ptr msg( new sensor_msgs::Imu( *msg_in ) );
    
    double timestamp = msg->header.stamp.toSec();
    m_mutex_buffer.lock();

    if ( m_last_timestamp_imu > 0.0 && timestamp < m_last_timestamp_imu ) {
        m_mutex_buffer.unlock();
        m_sig_buffer.notify_all();
        // ROS_ERROR( "imu loop back \n" );
        return;
    }
    // old 0.2
    if ( m_last_timestamp_imu > 0.0 && timestamp > m_last_timestamp_imu + 0.4 ) {
        m_mutex_buffer.unlock();
        m_sig_buffer.notify_all();
        // ROS_WARN( "imu time stamp Jumps %0.4lf seconds \n", timestamp - m_last_timestamp_imu );
        return;
    }

    m_last_timestamp_imu = timestamp;

    m_imu_buffer.push_back( msg );
    imu_buffer_vio.push_back( msg );
    m_mutex_buffer.unlock();
    m_sig_buffer.notify_all();
}

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

bool Voxel_mapping::sync_packages( LidarMeasureGroup &meas ) {
    if ( !m_imu_en ) {
        if ( !m_lidar_buffer.empty() ) {
            meas.lidar = m_lidar_buffer.front();
            meas.lidar_beg_time = m_time_buffer.front();
            m_lidar_buffer.pop_front();
            m_time_buffer.pop_front();
            return true;
        }
        return false;
    }

    if ( m_lidar_buffer.empty() || m_imu_buffer.empty() ) {
        return false;
    }

    /*** push a lidar scan ***/
    if ( !m_lidar_pushed ) {
        meas.lidar = m_lidar_buffer.front();
        if ( meas.lidar->points.size() <= 1 ) {
            m_lidar_buffer.pop_front();
            return false;
        }
        meas.lidar_beg_time = m_time_buffer.front();
        m_lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double( 1000 );
        m_lidar_pushed = true;
    }
    if ( m_last_timestamp_imu < m_lidar_end_time ) {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/

    // no img topic, means only has lidar topic
    if ( m_imu_en && m_last_timestamp_imu < m_lidar_end_time ) { // imu message needs to be larger than lidar_end_time, keep complete propagate.
        // ROS_ERROR("out sync");
        return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.
    if ( !m_imu_buffer.empty() ) {
        double imu_time = m_imu_buffer.front()->header.stamp.toSec();
        m.imu.clear();
        m_mutex_buffer.lock();
        while ( ( !m_imu_buffer.empty() && ( imu_time < m_lidar_end_time ) ) ) {
            imu_time = m_imu_buffer.front()->header.stamp.toSec();
            if ( imu_time > m_lidar_end_time )
                break;
            m.imu.push_back( m_imu_buffer.front() );
            m_imu_buffer.pop_front();
        }
    }
    m_lidar_buffer.pop_front();
    m_time_buffer.pop_front();
    m_mutex_buffer.unlock();
    m_sig_buffer.notify_all();
    m_lidar_pushed = false;   // sync one whole lidar scan.
    meas.is_lidar_end = true; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back( m );

    return true;
}

// 初始点云转换到当前帧所在位置
void Voxel_mapping::transformLidar( const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                                    pcl::PointCloud< pcl::PointXYZI >::Ptr &trans_cloud ) {
    trans_cloud->clear();
    for ( size_t i = 0; i < input_cloud->size(); i++ ) {
        pcl::PointXYZINormal p_c = input_cloud->points[ i ];
        Eigen::Vector3d      p( p_c.x, p_c.y, p_c.z );
        // p = rot * p + t;
        p = ( rot * ( m_extR * p + m_extT ) + t );
        pcl::PointXYZI pi;
        pi.x = p( 0 );
        pi.y = p( 1 );
        pi.z = p( 2 );
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back( pi );
    }
}

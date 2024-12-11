#include "voxel_mapping.hpp"
double g_LiDAR_frame_start_time = 0;

const bool intensity_contrast( PointType &x, PointType &y ) { return ( x.intensity > y.intensity ); };

const bool var_contrast( Point_with_var &x, Point_with_var &y ) { return ( x.m_var.diagonal().norm() < y.m_var.diagonal().norm() ); };

float calc_dist( PointType p1, PointType p2 ) {
    float d = ( p1.x - p2.x ) * ( p1.x - p2.x ) + ( p1.y - p2.y ) * ( p1.y - p2.y ) + ( p1.z - p2.z ) * ( p1.z - p2.z );
    return d;
}

void mapJet( double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b ) {
    r = 255;
    g = 255;
    b = 255;

    if ( v < vmin ) {
        v = vmin;
    }

    if ( v > vmax ) {
        v = vmax;
    }

    double dr, dg, db;

    if ( v < 0.1242 ) {
        db = 0.504 + ( ( 1. - 0.504 ) / 0.1242 ) * v;
        dg = dr = 0.;
    } else if ( v < 0.3747 ) {
        db = 1.;
        dr = 0.;
        dg = ( v - 0.1242 ) * ( 1. / ( 0.3747 - 0.1242 ) );
    } else if ( v < 0.6253 ) {
        db = ( 0.6253 - v ) * ( 1. / ( 0.6253 - 0.3747 ) );
        dg = 1.;
        dr = ( v - 0.3747 ) * ( 1. / ( 0.6253 - 0.3747 ) );
    } else if ( v < 0.8758 ) {
        db = 0.;
        dr = 1.;
        dg = ( 0.8758 - v ) * ( 1. / ( 0.8758 - 0.6253 ) );
    } else {
        db = 0.;
        dg = 0.;
        dr = 1. - ( v - 0.8758 ) * ( ( 1. - 0.504 ) / ( 1. - 0.8758 ) );
    }

    r = ( uint8_t )( 255 * dr );
    g = ( uint8_t )( 255 * dg );
    b = ( uint8_t )( 255 * db );
}

// 创建voxel,使用八叉树存储
void buildVoxelMap( const std::vector< Point_with_var > &input_points, const float voxel_size, const int max_layer,
                    const std::vector< int > &layer_init_num, const int max_points_size, const float planer_threshold,
                    std::unordered_map< VOXEL_LOC, OctoTree * > &feat_map ) {
    uint plsize = input_points.size();
    for ( uint i = 0; i < plsize; i++ ) {
        const Point_with_var p_v = input_points[ i ];
        float                loc_xyz[ 3 ];
        for ( int j = 0; j < 3; j++ ) {
            loc_xyz[ j ] = p_v.m_point[ j ] / voxel_size;
            if ( loc_xyz[ j ] < 0 ) {
                loc_xyz[ j ] -= 1.0;
            }
        }
        VOXEL_LOC position( ( int64_t ) loc_xyz[ 0 ], ( int64_t ) loc_xyz[ 1 ], ( int64_t ) loc_xyz[ 2 ] );
        auto      iter = feat_map.find( position );
        if ( iter != feat_map.end() ) {
            feat_map[ position ]->m_temp_points_.push_back( p_v );
            feat_map[ position ]->m_new_points_++;
        } else {
            OctoTree *octo_tree = new OctoTree( max_layer, 0, layer_init_num, max_points_size, planer_threshold );
            feat_map[ position ] = octo_tree;
            feat_map[ position ]->m_quater_length_ = voxel_size / 4;
            feat_map[ position ]->m_voxel_center_[ 0 ] = ( 0.5 + position.x ) * voxel_size;
            feat_map[ position ]->m_voxel_center_[ 1 ] = ( 0.5 + position.y ) * voxel_size;
            feat_map[ position ]->m_voxel_center_[ 2 ] = ( 0.5 + position.z ) * voxel_size;
            feat_map[ position ]->m_temp_points_.push_back( p_v );
            feat_map[ position ]->m_new_points_++;
            feat_map[ position ]->m_layer_init_num_ = layer_init_num;
        }
    }
    for ( auto iter = feat_map.begin(); iter != feat_map.end(); ++iter ) {
        iter->second->init_octo_tree();
    }
}

void BuildResidualListOMP( const unordered_map< VOXEL_LOC, OctoTree * > &voxel_map, const double voxel_size, const double sigma_num,
                           const int max_layer, const std::vector< Point_with_var > &pv_list, std::vector< ptpl > &ptpl_list,
                           std::vector< Eigen::Vector3d > &non_match ) {
    std::mutex mylock;
    ptpl_list.clear();
    std::vector< ptpl >   all_ptpl_list( pv_list.size() );
    std::vector< bool >   useful_ptpl( pv_list.size() );
    std::vector< size_t > index( pv_list.size() );
    for ( size_t i = 0; i < index.size(); ++i ) {
        index[ i ] = i;
        useful_ptpl[ i ] = false;
    }
    omp_set_num_threads( MP_PROC_NUM );
#pragma omp parallel for
    for ( int i = 0; i < index.size(); i++ ) {
        Point_with_var pv = pv_list[ i ];
        float          loc_xyz[ 3 ];
        for ( int j = 0; j < 3; j++ ) {
            loc_xyz[ j ] = pv.m_point_world[ j ] / voxel_size;
            if ( loc_xyz[ j ] < 0 ) {
                loc_xyz[ j ] -= 1.0;
            }
        }
        VOXEL_LOC position( ( int64_t ) loc_xyz[ 0 ], ( int64_t ) loc_xyz[ 1 ], ( int64_t ) loc_xyz[ 2 ] );
        auto      iter = voxel_map.find( position );
        if ( iter != voxel_map.end() ) {
            OctoTree *current_octo = iter->second;
            ptpl      single_ptpl;
            bool      is_sucess = false;
            double    prob = 0;
            build_single_residual( pv, current_octo, 0, max_layer, sigma_num, is_sucess, prob, single_ptpl );
            if ( !is_sucess ) {
                VOXEL_LOC near_position = position;
                if ( loc_xyz[ 0 ] > ( current_octo->m_voxel_center_[ 0 ] + current_octo->m_quater_length_ ) ) {
                    near_position.x = near_position.x + 1;
                }
                else if ( loc_xyz[ 0 ] < ( current_octo->m_voxel_center_[ 0 ] - current_octo->m_quater_length_ ) ) {
                    near_position.x = near_position.x - 1;
                }
                if ( loc_xyz[ 1 ] > ( current_octo->m_voxel_center_[ 1 ] + current_octo->m_quater_length_ ) ) {
                    near_position.y = near_position.y + 1;
                }
                else if ( loc_xyz[ 1 ] < ( current_octo->m_voxel_center_[ 1 ] - current_octo->m_quater_length_ ) ) {
                    near_position.y = near_position.y - 1;
                }
                if ( loc_xyz[ 2 ] > ( current_octo->m_voxel_center_[ 2 ] + current_octo->m_quater_length_ ) ) {
                    near_position.z = near_position.z + 1;
                }
                else if ( loc_xyz[ 2 ] < ( current_octo->m_voxel_center_[ 2 ] - current_octo->m_quater_length_ ) ) {
                    near_position.z = near_position.z - 1;
                }
                auto iter_near = voxel_map.find( near_position );
                if ( iter_near != voxel_map.end() ) {
                    build_single_residual( pv, iter_near->second, 0, max_layer, sigma_num, is_sucess, prob, single_ptpl );
                }
            }
            if ( is_sucess ) {
                mylock.lock();
                useful_ptpl[ i ] = true;
                all_ptpl_list[ i ] = single_ptpl;
                mylock.unlock();
            } else {
                mylock.lock();
                useful_ptpl[ i ] = false;
                mylock.unlock();
            }
        }
    }
    for ( size_t i = 0; i < useful_ptpl.size(); i++ ) {
        if ( useful_ptpl[ i ] ) {
            ptpl_list.push_back( all_ptpl_list[ i ] );
        }
    }
}

void build_single_residual( const Point_with_var &pv, const OctoTree *current_octo, const int current_layer, const int max_layer,
                            const double sigma_num, bool &is_sucess, double &prob, ptpl &single_ptpl ) {
    double          radius_k = 3;
    Eigen::Vector3d p_w = pv.m_point_world;
    if ( current_octo->m_plane_ptr_->m_is_plane ) {
        Plane &         plane = *current_octo->m_plane_ptr_;
        Eigen::Vector3d p_world_to_center = p_w - plane.m_center;
        float dis_to_plane = fabs( plane.m_normal( 0 ) * p_w( 0 ) + plane.m_normal( 1 ) * p_w( 1 ) + plane.m_normal( 2 ) * p_w( 2 ) + plane.m_d );
        float dis_to_center = ( plane.m_center( 0 ) - p_w( 0 ) ) * ( plane.m_center( 0 ) - p_w( 0 ) ) +
                              ( plane.m_center( 1 ) - p_w( 1 ) ) * ( plane.m_center( 1 ) - p_w( 1 ) ) +
                              ( plane.m_center( 2 ) - p_w( 2 ) ) * ( plane.m_center( 2 ) - p_w( 2 ) );
        float range_dis = sqrt( dis_to_center - dis_to_plane * dis_to_plane );

        if ( range_dis <= radius_k * plane.m_radius ) {
            Eigen::Matrix< double, 1, 6 > J_nq;
            J_nq.block< 1, 3 >( 0, 0 ) = p_w - plane.m_center;
            J_nq.block< 1, 3 >( 0, 3 ) = -plane.m_normal;
            double sigma_l = J_nq * plane.m_plane_var * J_nq.transpose();
            sigma_l += plane.m_normal.transpose() * pv.m_var * plane.m_normal;
            if ( dis_to_plane < sigma_num * sqrt( sigma_l ) ) {
                is_sucess = true;
                double this_prob = 1.0 / ( sqrt( sigma_l ) ) * exp( -0.5 * dis_to_plane * dis_to_plane / sigma_l );
                if ( this_prob > prob ) {
                    prob = this_prob;
                    single_ptpl.point = pv.m_point;
                    single_ptpl.plane_var = plane.m_plane_var;
                    single_ptpl.normal = plane.m_normal;
                    single_ptpl.center = plane.m_center;
                    single_ptpl.d = plane.m_d;
                    single_ptpl.layer = current_layer;
                }
                return;
            } else {
                return;
            }
        } else {
            return;
        }
    } else {
        if ( current_layer < max_layer ) {
            for ( size_t leafnum = 0; leafnum < 8; leafnum++ ) {
                if ( current_octo->m_leaves_[ leafnum ] != nullptr ) {
                    OctoTree *leaf_octo = current_octo->m_leaves_[ leafnum ];
                    build_single_residual( pv, leaf_octo, current_layer + 1, max_layer, sigma_num, is_sucess, prob, single_ptpl );
                }
            }
            return;
        } else {
            return;
        }
    }
}

void updateVoxelMap( const std::vector< Point_with_var > &input_points, const float voxel_size, const int max_layer,
                     const std::vector< int > &layer_init_num, const int max_points_size, const float planer_threshold,
                     std::unordered_map< VOXEL_LOC, OctoTree * > &feat_map ) {
    uint plsize = input_points.size();
    for ( uint i = 0; i < plsize; i++ ) {
        const Point_with_var p_v = input_points[ i ];
        float                loc_xyz[ 3 ];
        for ( int j = 0; j < 3; j++ ) {
            loc_xyz[ j ] = p_v.m_point[ j ] / voxel_size;
            if ( loc_xyz[ j ] < 0 ) {
                loc_xyz[ j ] -= 1.0;
            }
        }
        VOXEL_LOC position( ( int64_t ) loc_xyz[ 0 ], ( int64_t ) loc_xyz[ 1 ], ( int64_t ) loc_xyz[ 2 ] );
        auto      iter = feat_map.find( position );
        if ( iter != feat_map.end() ) {
            feat_map[ position ]->UpdateOctoTree( p_v );
        } else {
            OctoTree *octo_tree = new OctoTree( max_layer, 0, layer_init_num, max_points_size, planer_threshold );
            feat_map[ position ] = octo_tree;
            feat_map[ position ]->m_quater_length_ = voxel_size / 4;
            feat_map[ position ]->m_voxel_center_[ 0 ] = ( 0.5 + position.x ) * voxel_size;
            feat_map[ position ]->m_voxel_center_[ 1 ] = ( 0.5 + position.y ) * voxel_size;
            feat_map[ position ]->m_voxel_center_[ 2 ] = ( 0.5 + position.z ) * voxel_size;
            feat_map[ position ]->UpdateOctoTree( p_v );
        }
    }
}

void calcBodyVar( Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &var ) {
    if ( pb[ 2 ] == 0 )
        pb[ 2 ] = 0.0001;
    float           range = sqrt( pb[ 0 ] * pb[ 0 ] + pb[ 1 ] * pb[ 1 ] + pb[ 2 ] * pb[ 2 ] );
    float           range_var = range_inc * range_inc;
    Eigen::Matrix2d direction_var;
    direction_var << pow( sin( DEG2RAD( degree_inc ) ), 2 ), 0, 0, pow( sin( DEG2RAD( degree_inc ) ), 2 );
    Eigen::Vector3d direction( pb );
    direction.normalize();
    Eigen::Matrix3d direction_hat;
    direction_hat << 0, -direction( 2 ), direction( 1 ), direction( 2 ), 0, -direction( 0 ), -direction( 1 ), direction( 0 ), 0;
    Eigen::Vector3d base_vector1( 1, 1, -( direction( 0 ) + direction( 1 ) ) / direction( 2 ) );
    base_vector1.normalize();
    Eigen::Vector3d base_vector2 = base_vector1.cross( direction );
    base_vector2.normalize();
    Eigen::Matrix< double, 3, 2 > N;
    N << base_vector1( 0 ), base_vector2( 0 ), base_vector1( 1 ), base_vector2( 1 ), base_vector1( 2 ), base_vector2( 2 );
    Eigen::Matrix< double, 3, 2 > A = range * direction_hat * N;
    var = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
};

// 创建voxel并初始化
bool Voxel_mapping::voxel_map_init() {
    pcl::PointCloud< pcl::PointXYZI >::Ptr world_lidar( new pcl::PointCloud< pcl::PointXYZI > );
    Eigen::Quaterniond                     q( state.rot_end );

    // std::cout << "Begin build unorder map" << std::endl;
    transformLidar( state.rot_end, state.pos_end, m_feats_undistort, world_lidar );
    std::vector< Point_with_var > pv_list;
    pv_list.reserve( world_lidar->size() );
    for ( size_t i = 0; i < world_lidar->size(); i++ ) {
        Point_with_var pv;
        pv.m_point << world_lidar->points[ i ].x, world_lidar->points[ i ].y, world_lidar->points[ i ].z;
        V3D point_this( m_feats_undistort->points[ i ].x, m_feats_undistort->points[ i ].y, m_feats_undistort->points[ i ].z );
        M3D var;
        calcBodyVar( point_this, m_dept_err, m_beam_err, var );

        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX( point_this );
        var = state.rot_end * var * state.rot_end.transpose() +
              ( -point_crossmat ) * state.cov.block< 3, 3 >( 0, 0 ) * ( -point_crossmat ).transpose() + state.cov.block< 3, 3 >( 3, 3 );
        pv.m_var = var;
        pv_list.push_back( pv );
    }

    buildVoxelMap( pv_list, m_max_voxel_size, m_max_layer, m_layer_init_size, m_max_points_size, m_min_eigen_value, m_feat_map );

    return true;
}

/*** EKF update ***/
void Voxel_mapping::lio_state_estimation( StatesGroup &state_propagat ) {
    m_point_selected_surf.resize( m_feats_down_size, true );
    m_pointSearchInd_surf.resize( m_feats_down_size );
    m_Nearest_Points.resize( m_feats_down_size );
    m_cross_mat_list.clear();
    m_cross_mat_list.reserve( m_feats_down_size );
    m_body_cov_list.clear();
    m_body_cov_list.reserve( m_feats_down_size );

    if ( m_use_new_map ) {
        if ( m_feat_map.empty() ) {
            printf( "feat_map.empty!!" );
            return;
        }

        for ( size_t i = 0; i < m_feats_down_body->size(); i++ ) {
            V3D point_this( m_feats_down_body->points[ i ].x, m_feats_down_body->points[ i ].y, m_feats_down_body->points[ i ].z );
            if ( point_this[ 2 ] == 0 ) {
                point_this[ 2 ] = 0.001;
            }
            M3D var;
            calcBodyVar( point_this, m_dept_err, m_beam_err, var );
            m_body_cov_list.push_back( var );
            point_this = m_extR * point_this + m_extT;
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX( point_this );
            m_cross_mat_list.push_back( point_crossmat );
        }
    }

    if ( !m_flg_EKF_inited )
        return;

    m_point_selected_surf.resize( m_feats_down_size, true );
    m_Nearest_Points.resize( m_feats_down_size );
    int  rematch_num = 0;
    bool nearest_search_en = true; //
    MD( DIM_STATE, DIM_STATE ) G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    for ( int iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++ ) {
        m_laserCloudOri->clear(); //
        m_corr_normvect->clear(); //
        m_total_residual = 0.0;   //

        std::vector< double > r_list;
        std::vector< ptpl >   ptpl_list;
        // ptpl_list.reserve( m_feats_down_size );
        if ( m_use_new_map ) {
            vector< Point_with_var >               pv_list;
            pcl::PointCloud< pcl::PointXYZI >::Ptr world_lidar( new pcl::PointCloud< pcl::PointXYZI > );
            transformLidar( state.rot_end, state.pos_end, m_feats_down_body, world_lidar );

            M3D rot_var = state.cov.block< 3, 3 >( 0, 0 );
            M3D t_var = state.cov.block< 3, 3 >( 3, 3 );
            for ( size_t i = 0; i < m_feats_down_body->size(); i++ ) {
                Point_with_var pv;
                pv.m_point << m_feats_down_body->points[ i ].x, m_feats_down_body->points[ i ].y, m_feats_down_body->points[ i ].z;
                pv.m_point_world << world_lidar->points[ i ].x, world_lidar->points[ i ].y, world_lidar->points[ i ].z;
                M3D cov = m_body_cov_list[ i ];
                M3D point_crossmat = m_cross_mat_list[ i ];

                cov = state.rot_end * cov * state.rot_end.transpose() + ( -point_crossmat ) * rot_var * ( -point_crossmat.transpose() ) + t_var;
                pv.m_var = cov;
                pv_list.push_back( pv );
            }

            auto               scan_match_time_start = std::chrono::high_resolution_clock::now();
            std::vector< V3D > non_match_list;
            BuildResidualListOMP( m_feat_map, m_max_voxel_size, 3.0, m_max_layer, pv_list, ptpl_list, non_match_list );
            m_effct_feat_num = 0;
            int layer_match[ 4 ] = { 0 };

            for ( int i = 0; i < ptpl_list.size(); i++ ) {
                PointType pi_body;
                PointType pi_world;
                PointType pl;
                pi_body.x = ptpl_list[ i ].point( 0 );
                pi_body.y = ptpl_list[ i ].point( 1 );
                pi_body.z = ptpl_list[ i ].point( 2 );
                Eigen::Vector3d point_world;
                pointBodyToWorld( ptpl_list[ i ].point, point_world );
                pl.x = ptpl_list[ i ].normal( 0 );
                pl.y = ptpl_list[ i ].normal( 1 );
                pl.z = ptpl_list[ i ].normal( 2 );
                m_effct_feat_num++;
                float dis = ( point_world[ 0 ] * pl.x + point_world[ 1 ] * pl.y + point_world[ 2 ] * pl.z + ptpl_list[ i ].d );
                pl.intensity = dis;
                m_laserCloudOri->push_back( pi_body );
                m_corr_normvect->push_back( pl );
                m_total_residual += fabs( dis );
                layer_match[ ptpl_list[ i ].layer ]++;
            }
            auto scan_match_time_end = std::chrono::high_resolution_clock::now();
            m_res_mean_last = m_total_residual / m_effct_feat_num;
        } else {
/** Old map ICP **/
#ifdef MP_EN
            omp_set_num_threads( 10 );
#pragma omp parallel for
#endif
            for ( int i = 0; i < m_feats_down_size; i++ ) {
                PointType &point_body = m_feats_down_body->points[ i ];
                PointType &point_world = m_feats_down_world->points[ i ];
                V3D        p_body( point_body.x, point_body.y, point_body.z );
                /* transform to world frame */
                pointBodyToWorld( point_body, point_world );
                vector< float > pointSearchSqDis( NUM_MATCH_POINTS );
#ifdef USE_ikdtree
                auto &points_near = m_Nearest_Points[ i ];
#else
                auto &points_near = pointSearchInd_surf[ i ];
#endif
                uint8_t search_flag = 0;
                double  search_start = omp_get_wtime();
                if ( nearest_search_en ) {
/** Find the closest surfaces in the map **/
#ifdef USE_ikdtree
#ifdef USE_ikdforest
                    search_flag = ikdforest.Nearest_Search( point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, first_lidar_time, 5 );
#else
                    m_ikdtree.Nearest_Search( point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis );
#endif
#else
                    kdtreeSurfFromMap->nearestKSearch( point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis );
#endif

                    m_point_selected_surf[ i ] = pointSearchSqDis[ NUM_MATCH_POINTS - 1 ] > 5 ? false : true;

#ifdef USE_ikdforest
                    point_selected_surf[ i ] = point_selected_surf[ i ] && ( search_flag == 0 );
#endif
                    m_kdtree_search_time += omp_get_wtime() - search_start;
                    m_kdtree_search_counter++;
                }

                if ( !m_point_selected_surf[ i ] || points_near.size() < NUM_MATCH_POINTS )
                    continue;

                VF( 4 ) pabcd;
                m_point_selected_surf[ i ] = false;
                if ( esti_plane( pabcd, points_near, 0.05f ) ) //(planeValid)
                {
                    float pd2 = pabcd( 0 ) * point_world.x + pabcd( 1 ) * point_world.y + pabcd( 2 ) * point_world.z + pabcd( 3 );
                    float s = 1 - 0.9 * fabs( pd2 ) / sqrt( p_body.norm() );

                    if ( s > 0.9 )
                    {
                        m_point_selected_surf[ i ] = true;
                        m_normvec->points[ i ].x = pabcd( 0 );
                        m_normvec->points[ i ].y = pabcd( 1 );
                        m_normvec->points[ i ].z = pabcd( 2 );
                        m_normvec->points[ i ].intensity = pd2;
                        m_res_last[ i ] = abs( pd2 );
                    }
                }
            }
            // cout<<"pca time test: "<<pca_time1<<" "<<pca_time2<<endl;
            m_effct_feat_num = 0;

            for ( int i = 0; i < m_feats_down_size; i++ )
            {
                if ( m_point_selected_surf[ i ] && ( m_res_last[ i ] <= 2.0 ) )
                {
                    m_laserCloudOri->points[ m_effct_feat_num ] = m_feats_down_body->points[ i ];
                    m_corr_normvect->points[ m_effct_feat_num ] = m_normvec->points[ i ];
                    m_total_residual += m_res_last[ i ];
                    m_effct_feat_num++;
                }
            }

            m_res_mean_last = m_total_residual / m_effct_feat_num;
        }

        double solve_start = omp_get_wtime();
        /*** Computation of Measuremnt Jacobian matrix H and measurents covarience
         * ***/
        MatrixXd Hsub( m_effct_feat_num, 6 );
        MatrixXd Hsub_T_R_inv( 6, m_effct_feat_num );
        VectorXd R_inv( m_effct_feat_num );
        VectorXd meas_vec( m_effct_feat_num );
        meas_vec.setZero();

        for ( int i = 0; i < m_effct_feat_num; i++ ) {
            const PointType &laser_p = m_laserCloudOri->points[ i ];
            V3D              point_this( laser_p.x, laser_p.y, laser_p.z );
            V3D              point_body( laser_p.x, laser_p.y, laser_p.z );
            point_this = m_extR * point_this + m_extT;
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX( point_this );

            /*** get the normal vector of closest surface/corner ***/
            PointType &norm_p = m_corr_normvect->points[ i ];
            V3D        norm_vec( norm_p.x, norm_p.y, norm_p.z );

            if ( m_use_new_map ) {
                V3D point_world = state.rot_end * point_this + state.pos_end;
                // /*** get the normal vector of closest surface/corner ***/

                M3D var;

                if ( m_p_pre->calib_laser ) {
                    calcBodyVar( point_this, m_dept_err, CALIB_ANGLE_COV, var );
                } else {
                    calcBodyVar( point_this, m_dept_err, m_beam_err, var );
                }
                var = state.rot_end * m_extR * var * ( state.rot_end * m_extR ).transpose();
                // bug exist, to be fixed
                Eigen::Matrix< double, 1, 6 > J_nq;
                J_nq.block< 1, 3 >( 0, 0 ) = point_world - ptpl_list[ i ].center;
                J_nq.block< 1, 3 >( 0, 3 ) = -ptpl_list[ i ].normal;
                double sigma_l = J_nq * ptpl_list[ i ].plane_var * J_nq.transpose();
                R_inv( i ) = 1.0 / ( sigma_l + norm_vec.transpose() * var * norm_vec );
            } else {
                R_inv( i ) = 1 / LASER_POINT_COV;
            }
            m_laserCloudOri->points[ i ].intensity = sqrt( R_inv( i ) );

            /*** calculate the Measuremnt Jacobian matrix H ***/
            V3D A( point_crossmat * state.rot_end.transpose() * norm_vec );
            Hsub.row( i ) << VEC_FROM_ARRAY( A ), norm_p.x, norm_p.y, norm_p.z;
            Hsub_T_R_inv.col( i ) << A[ 0 ] * R_inv( i ), A[ 1 ] * R_inv( i ), A[ 2 ] * R_inv( i ), norm_p.x * R_inv( i ), norm_p.y * R_inv( i ),
                norm_p.z * R_inv( i );
            /*** Measuremnt: distance to the closest surface/corner ***/
            meas_vec( i ) = -norm_p.intensity;
        }
        m_solve_const_H_time += omp_get_wtime() - solve_start;

        m_EKF_stop_flg = false;
        m_flg_EKF_converged = false;

        MatrixXd K( DIM_STATE, m_effct_feat_num );

        /*** Iterative Kalman Filter Update ***/
        // auto &&Hsub_T = Hsub.transpose();
        auto &&HTz = Hsub_T_R_inv * meas_vec;
        H_T_H.block< 6, 6 >( 0, 0 ) = Hsub_T_R_inv * Hsub;
        // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6,6>(0,0));
        MD( DIM_STATE, DIM_STATE ) &&K_1 = ( H_T_H + state.cov.inverse() ).inverse();
        G.block< DIM_STATE, 6 >( 0, 0 ) = K_1.block< DIM_STATE, 6 >( 0, 0 ) * H_T_H.block< 6, 6 >( 0, 0 );
        auto vec = state_propagat - state;
        VD( DIM_STATE )
        solution = K_1.block< DIM_STATE, 6 >( 0, 0 ) * HTz + vec - G.block< DIM_STATE, 6 >( 0, 0 ) * vec.block< 6, 1 >( 0, 0 );

        int minRow, minCol;

        state += solution;

        auto rot_add = solution.block< 3, 1 >( 0, 0 );
        auto t_add = solution.block< 3, 1 >( 3, 0 );

        if ( ( rot_add.norm() * 57.3 < 0.01 ) && ( t_add.norm() * 100 < 0.015 ) ) {
            m_flg_EKF_converged = true;
        }

        m_euler_cur = RotMtoEuler( state.rot_end );

        /*** Rematch Judgement ***/
        nearest_search_en = false;
        if ( m_flg_EKF_converged || ( ( rematch_num == 0 ) && ( iterCount == ( NUM_MAX_ITERATIONS - 2 ) ) ) ) {
            nearest_search_en = true;
            rematch_num++;
        }

        /*** Convergence Judgements and Covariance Update ***/
        if ( !m_EKF_stop_flg && ( rematch_num >= 2 || ( iterCount == NUM_MAX_ITERATIONS - 1 ) ) ) {
            /*** Covariance Update ***/
            state.cov = ( I_STATE - G ) * state.cov;
            m_total_distance += ( state.pos_end - m_position_last ).norm();
            m_position_last = state.pos_end;
            m_geo_Quat = tf::createQuaternionMsgFromRollPitchYaw( m_euler_cur( 0 ), m_euler_cur( 1 ), m_euler_cur( 2 ) );

            m_EKF_stop_flg = true;
        }
        m_solve_time += omp_get_wtime() - solve_start;

        if ( m_EKF_stop_flg )
            break;
    }
}

void Voxel_mapping::init_ros_node() {
    m_ros_node_ptr = std::make_shared< ros::NodeHandle >();
}

int Voxel_mapping::service_LiDAR_update() {
    m_pcl_wait_pub->clear();

    ros::Subscriber sub_pcl;
    sub_pcl = m_ros_node_ptr->subscribe( m_lid_topic, 200000, &Voxel_mapping::standard_pcl_cbk, this );

    ros::Subscriber sub_imu = m_ros_node_ptr->subscribe( m_imu_topic, 200000, &Voxel_mapping::imu_cbk, this );
    
#ifndef USE_IKFOM
    VD( DIM_STATE ) solution;
    MD( DIM_STATE, DIM_STATE ) G, H_T_H, I_STATE;
    V3D         rot_add, t_add;
    StatesGroup state_propagat;
    PointType   pointOri, pointSel, coeff;
#endif
    int    effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_solve = 0, aver_time_const_H_time = 0;

    FOV_DEG = ( m_fov_deg + 10.0 ) > 179.9 ? 179.9 : ( m_fov_deg + 10.0 );
    HALF_FOV_COS = cos( ( FOV_DEG ) *0.5 * PI_M / 180.0 );

    m_downSizeFilterSurf.setLeafSize( m_filter_size_surf_min, m_filter_size_surf_min, m_filter_size_surf_min );

#ifdef USE_ikdforest
    ikdforest.Set_balance_criterion_param( 0.6 );
    ikdforest.Set_delete_criterion_param( 0.5 );
    ikdforest.Set_environment( laserCloudDepth, laserCloudWidth, laserCloudHeight, cube_len );
    ikdforest.Set_downsample_param( filter_size_map_min );
#endif

    shared_ptr< ImuProcess > p_imu( new ImuProcess() );
    m_imu_process = p_imu;
    m_extT << VEC_FROM_ARRAY( m_extrin_T );
    m_extR << MAT_FROM_ARRAY( m_extrin_R );

    p_imu->set_extrinsic( m_extT, m_extR );
    p_imu->set_gyr_cov_scale( V3D( m_gyr_cov, m_gyr_cov, m_gyr_cov ) );
    p_imu->set_acc_cov_scale( V3D( m_acc_cov, m_acc_cov, m_acc_cov ) );
    p_imu->set_gyr_bias_cov( V3D( 0.0001, 0.0001, 0.0001 ) );
    p_imu->set_acc_bias_cov( V3D( 0.0001, 0.0001, 0.0001 ) );
    p_imu->set_imu_init_frame_num( m_imu_int_frame );

    if ( !m_imu_en )
        p_imu->disable_imu();

#ifndef USE_IKFOM
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();
#endif

#ifdef USE_IKFOM
    double epsi[ 23 ] = { 0.001 };
    fill( epsi, epsi + 23, 0.001 );
    kf.init_dyn_share( get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi );
#endif

#ifdef USE_ikdforest
    ikdforest.Set_balance_criterion_param( 0.6 );
    ikdforest.Set_delete_criterion_param( 0.5 );
#endif
    ros::Rate rate( 5000 );
    bool      status = ros::ok();
    while ( ( status = ros::ok() ) ) {
        if ( m_flg_exit )
            break;
        ros::spinOnce();
        if ( !sync_packages( m_Lidar_Measures ) ) {
            status = ros::ok();
            rate.sleep();
            continue;
        }

        /*** Packaged got ***/
        if ( m_flg_reset ) {
            ROS_WARN( "reset when rosbag play back" );
            p_imu->Reset();
            m_flg_reset = false;
            continue;
        }

        if ( !m_is_first_frame ) {
            m_first_lidar_time = m_Lidar_Measures.lidar_beg_time;
            p_imu->first_lidar_time = m_first_lidar_time;
            m_is_first_frame = true;
            cout << "FIRST LIDAR FRAME!" << endl;
        }
        double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

        m_match_time = m_kdtree_search_time = m_kdtree_search_counter = m_solve_time = m_solve_const_H_time = svd_time = 0;
        t0 = omp_get_wtime();
        g_LiDAR_frame_start_time = t0;
        auto t_all_begin = std::chrono::high_resolution_clock::now();
#ifdef USE_IKFOM
        p_imu->Process( LidarMeasures, kf, feats_undistort );
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
#else
        m_mutex_lio_process.lock();
        p_imu->Process2( m_Lidar_Measures, state,         m_feats_undistort );
        m_mutex_lio_process.unlock();
        // ANCHOR: Remove point that nearing 10 meter
        if ( 0 ) {
            int feats_undistort_available = 0;
            for ( int i = 0; i < m_feats_undistort->size(); i++ ) {
                vec_3 pt_vec = vec_3( m_feats_undistort->points[ i ].x, m_feats_undistort->points[ i ].y, m_feats_undistort->points[ i ].z );
                if ( pt_vec.norm() <= 10 ) {
                    continue;
                }
                feats_undistort_available++;
                m_feats_undistort->points[ feats_undistort_available ] = m_feats_undistort->points[ i ];
            }
            m_feats_undistort->points.resize( feats_undistort_available );
        }
        
        state_propagat = state;
        // FILE *fp = fopen(("/home/" + std::to_string(frame_num) + "state.txt").c_str(), "w+");
        // if (fp)
        // {
        //     auto q = eigen_q(state.rot_end);
        //     fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf\r\n", q.w(), q.x(), q.y(), q.z(), 
        //     state.pos_end(0), state.pos_end(1), state.pos_end(2));
        //     fclose(fp);
        // }
        
#endif
        if ( m_p_pre->calib_laser ) {
            for ( size_t i = 0; i < m_feats_undistort->size(); i++ ) {
                PointType pi = m_feats_undistort->points[ i ];
                double    range = sqrt( pi.x * pi.x + pi.y * pi.y + pi.z * pi.z );
                double    calib_vertical_angle = deg2rad( 0.15 );
                double    vertical_angle = asin( pi.z / range ) + calib_vertical_angle;
                double    horizon_angle = atan2( pi.y, pi.x );
                pi.z = range * sin( vertical_angle );
                double project_len = range * cos( vertical_angle );
                pi.x = project_len * cos( horizon_angle );
                pi.y = project_len * sin( horizon_angle );
                m_feats_undistort->points[ i ] = pi;
            }
        }

        if ( m_feats_undistort->empty() || ( m_feats_undistort == nullptr ) ) {
            cout << " No point!!!" << endl;
            continue;
        }

        m_flg_EKF_inited = ( m_Lidar_Measures.lidar_beg_time - m_first_lidar_time ) < INIT_TIME ? false : true;

        if ( !m_use_new_map)
            laser_map_fov_segment();

        /*** downsample the feature points in a scan ***/
        // pcl::io::savePLYFile("/home/" + std::to_string(frame_num) + "_origin_frame.ply", *m_feats_undistort);

        m_downSizeFilterSurf.setInputCloud( m_feats_undistort );
        m_downSizeFilterSurf.filter( *m_feats_down_body );
        m_feats_down_size = m_feats_down_body->points.size();

        /*** initialize the map ***/
        if ( m_use_new_map ) {
            if ( !m_init_map ) {
                m_init_map = voxel_map_init();
                frame_num++;
                continue;
            };
        } else if ( m_ikdtree.Root_Node == nullptr ) {
            if ( m_feats_down_body->points.size() > 5 ) {
                m_ikdtree.set_downsample_param( m_filter_size_map_min );
                frameBodyToWorld( m_feats_down_body, m_feats_down_world );
                m_ikdtree.Build( m_feats_down_world->points );
            }
            continue;
        }
        int featsFromMapNum = m_ikdtree.size();

        /*** ICP and iterated Kalman filter update ***/
        m_normvec->resize( m_feats_down_size );
        m_feats_down_world->clear();
        m_feats_down_world->resize( m_feats_down_size );
        // vector<double> res_last(feats_down_size, 1000.0); // initial //
        m_res_last.resize( m_feats_down_size, 1000.0 );
        m_point_selected_surf.resize( m_feats_down_size, true );
        m_pointSearchInd_surf.resize( m_feats_down_size );
        m_Nearest_Points.resize( m_feats_down_size );

        t1 = omp_get_wtime();
        t2 = omp_get_wtime();

/*** iterated state estimation ***/
        double t_update_start = omp_get_wtime();
#ifdef USE_IKFOM
        double solve_H_time = 0;
        kf.update_iterated_dyn_share_modified( LASER_POINT_COV, solve_H_time );
        // state_ikfom updated_state = kf.get_x();
        state_point = kf.get_x();
        // euler_cur = RotMtoEuler(state_point.rot.toRotationMatrix());
        euler_cur = SO3ToEuler( state_point.rot );
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        geoQuat.x = state_point.rot.coeffs()[ 0 ];
        geoQuat.y = state_point.rot.coeffs()[ 1 ];
        geoQuat.z = state_point.rot.coeffs()[ 2 ];
        geoQuat.w = state_point.rot.coeffs()[ 3 ];
#else

        if ( m_lidar_en ) {
            lio_state_estimation( state_propagat );
        }
#endif
        m_euler_cur = RotMtoEuler( state.rot_end );
        m_geo_Quat = tf::createQuaternionMsgFromRollPitchYaw( m_euler_cur( 0 ), m_euler_cur( 1 ), m_euler_cur( 2 ) );

        if ( m_lidar_en)
            map_incremental_grow();

        frame_num++;
        if ( m_lidar_en ) {
            m_euler_cur = RotMtoEuler( state.rot_end );
        }
    }

    return 0;
}

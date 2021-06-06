void Transformer::lookupTwist(const std::string &tracking_frame, const std::string &observation_frame, const std::string &reference_frame,
                              const tf::Point &reference_point, const std::string &reference_point_frame,
                              const ros::Time &time, const ros::Duration &averaging_interval,
                              geometry_msgs::Twist &twist) const

{
    ros::Time latest_time, target_time;
    getLatestCommonTime(observation_frame, tracking_frame, latest_time, NULL);
    if (ros::Time() == time)
        target_time = latest_time;
    else
        target_time = time;
    ros::Time end_time = std::min(target_time + averaging_interval * 0.5, latest_time);
    ros::Time start_time = std::max(ros::Time().fromSec(.00001) + averaging_interval, end_time) - averaging_interval; // don't collide with zero
    ros::Duration corrected_averaging_interval = end_time - start_time;                                               //correct for the possiblity that start time was truncated above.
    StampedTransform start, end;
    lookupTransform(observation_frame, tracking_frame, start_time, start);
    lookupTransform(observation_frame, tracking_frame, end_time, end);
    tf::Matrix3x3 temp = start.getBasis().inverse() * end.getBasis();
    tf::Quaternion quat_temp;
    temp.getRotation(quat_temp);
    tf::Vector3 o = start.getBasis() * quat_temp.getAxis();
    tfScalar ang = quat_temp.getAngle();
    double delta_x = end.getOrigin().getX() - start.getOrigin().getX();
    double delta_y = end.getOrigin().getY() - start.getOrigin().getY();
    double delta_z = end.getOrigin().getZ() - start.getOrigin().getZ();
    tf::Vector3 twist_vel((delta_x) / corrected_averaging_interval.toSec(),
                          (delta_y) / corrected_averaging_interval.toSec(),
                          (delta_z) / corrected_averaging_interval.toSec());
    tf::Vector3 twist_rot = o * (ang / corrected_averaging_interval.toSec());
    // This is a twist w/ reference frame in observation_frame  and reference point is in the tracking_frame at the origin (at start_time)

    //correct for the position of the reference frame
    tf::StampedTransform inverse;
    lookupTransform(reference_frame, tracking_frame, target_time, inverse);
    tf::Vector3 out_rot = inverse.getBasis() * twist_rot;
    tf::Vector3 out_vel = inverse.getBasis() * twist_vel + inverse.getOrigin().cross(out_rot);

    //Rereference the twist about a new reference point
    // Start by computing the original reference point in the reference frame:
    tf::Stamped<tf::Point>
        rp_orig(tf::Point(0, 0, 0), target_time, tracking_frame);
    transformPoint(reference_frame, rp_orig, rp_orig);
    // convert the requrested reference point into the right frame
    tf::Stamped<tf::Point>
        rp_desired(reference_point, target_time, reference_point_frame);
    transformPoint(reference_frame, rp_desired, rp_desired);
    // compute the delta
    tf::Point delta = rp_desired - rp_orig;
    // Correct for the change in reference point
    out_vel = out_vel + out_rot * delta;
    // out_rot unchanged
    twist.linear.x = out_vel.x();
    twist.linear.y = out_vel.y();
    twist.linear.z = out_vel.z();
    twist.angular.x = out_rot.x();
    twist.angular.y = out_rot.y();
    twist.angular.z = out_rot.z();
};
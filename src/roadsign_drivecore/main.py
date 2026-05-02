import carla
import random
import time
import pygame
import numpy as np
import math
import threading

# 核心配置（新增红绿灯专用参数）
CONFIG = {
    "CARLA_HOST": "localhost",
    "CARLA_PORT": 2000,
    "CAMERA_WIDTH": 800,
    "CAMERA_HEIGHT": 600,
    "CRUISE_SPEED": 40,
    "INTERSECTION_SPEED": 25,
    "SAFE_STOP_DISTANCE": 15,
    "MIN_STOP_DISTANCE": 3,
    # ========== 红绿灯专用配置 ==========
    "MAX_TRAFFIC_LIGHT_PREVIEW": 100,  # 提前100米开始检测红绿灯
    "FRICTION_COEFFICIENT": 0.7,  # 路面摩擦系数（沥青路面标准值）
    "GRAVITY": 9.8,  # 重力加速度
    "YELLOW_SLOWDOWN_DISTANCE": 40  # 黄灯40米外开始减速
}

# 全局状态与线程锁（保留线程安全修复）
need_vehicle_reset = False
g_thread_lock = threading.Lock()


# 初始化Pygame显示
def init_pygame(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA V5.0 零闯红灯稳定版")
    return display


# 转换CARLA图像用于显示
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    return array


# 获取当前车速 km/h
def get_speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6


# 计算转向角
def get_steer(vehicle_transform, waypoint_transform):
    v_loc = vehicle_transform.location
    v_forward = vehicle_transform.get_forward_vector()
    wp_loc = waypoint_transform.location

    direction = carla.Vector3D(wp_loc.x - v_loc.x, wp_loc.y - v_loc.y, 0.0)
    v_forward = carla.Vector3D(v_forward.x, v_forward.y, 0.0)

    dir_norm = math.hypot(direction.x, direction.y)
    fwd_norm = math.hypot(v_forward.x, v_forward.y)
    if dir_norm < 1e-5 or fwd_norm < 1e-5:
        return 0.0

    dot = (v_forward.x * direction.x + v_forward.y * direction.y) / (dir_norm * fwd_norm)
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)
    cross = v_forward.x * direction.y - v_forward.y * direction.x
    if cross < 0:
        angle *= -1
    return max(-1.0, min(1.0, angle * 2.0))


# ========== V5.0 核心修复：精准计算到停止线的距离+提前获取红绿灯 ==========
def get_stop_line_info(vehicle, map):
    """
    返回：(到停止线的距离, 是否已进入路口, 前方路口的红绿灯对象)
    解决核心问题：提前检测红绿灯，精准定位停止线位置
    """
    vehicle_loc = vehicle.get_transform().location
    current_wp = map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    # 先判断当前是否已经在路口内
    if current_wp.is_junction:
        return 0.0, True, None

    total_distance = 0.0
    check_wp = current_wp
    traffic_light = None

    # 向前遍历路点，直到找到路口/停止线，或达到最大检测距离
    for _ in range(int(CONFIG["MAX_TRAFFIC_LIGHT_PREVIEW"] // 2)):
        next_wps = check_wp.next(2.0)
        if not next_wps:
            break

        # 路口分叉优先选直行（同车道ID，避免选到转弯车道）
        next_wp = next_wps[0]
        for wp in next_wps:
            if wp.lane_id == check_wp.lane_id:
                next_wp = wp
                break

        # 检测到路口入口
        if next_wp.is_junction:
            # 提前获取该路口对应车道的红绿灯（核心修复：不等进入触发区）
            junction = next_wp.get_junction()
            if junction:
                try:
                    # 获取当前车道在路口的红绿灯
                    traffic_light = vehicle.get_traffic_light()
                    # 兜底：如果当前没拿到，从路口的红绿灯列表里匹配
                    if not traffic_light:
                        tl_list = junction.get_traffic_lights()
                        for tl in tl_list:
                            trigger_volume = tl.get_trigger_volume()
                            # 判断当前车道是否在该红绿灯的管控范围内
                            if trigger_volume.contains(vehicle_loc, vehicle.get_transform()):
                                traffic_light = tl
                                break
                except:
                    traffic_light = None
            return total_distance, False, traffic_light

        check_wp = next_wp
        total_distance += 2.0

    # 前方无路口
    return 999.0, False, None


# ========== V5.0 物理级刹车距离计算 ==========
def calc_required_brake_distance(speed_kmh):
    """根据当前车速，计算完全刹停所需的最小距离（物理公式）"""
    if speed_kmh < 1:
        return 0.0
    speed_ms = speed_kmh / 3.6  # 转成m/s
    # 刹车距离公式：v²/(2*μ*g) + 安全余量
    brake_distance = (speed_ms ** 2) / (2 * CONFIG["FRICTION_COEFFICIENT"] * CONFIG["GRAVITY"])
    return brake_distance + CONFIG["SAFE_STOP_DISTANCE"]


# 碰撞事件回调（保留线程安全）
def on_collision(event):
    global need_vehicle_reset
    with g_thread_lock:
        if not need_vehicle_reset:
            need_vehicle_reset = True
            collision_force = event.normal_impulse.length()
            print(f"【碰撞保护】检测到碰撞！强度：{collision_force:.1f}，准备重置车辆")


def main():
    global need_vehicle_reset
    actor_list = []
    image_surface = [None]

    try:
        # 连接CARLA
        client = carla.Client(CONFIG["CARLA_HOST"], CONFIG["CARLA_PORT"])
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        # 生成主车（带重试机制）
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        vehicle = None
        for retry in range(5):
            try:
                spawn_point = random.choice(map.get_spawn_points())
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                break
            except Exception as e:
                print(f"主车生成失败，重试 {retry + 1}/5: {e}")
                time.sleep(0.5)

        if vehicle is None:
            raise RuntimeError("无法生成主车，请检查CARLA服务端状态")
        actor_list.append(vehicle)
        print("主车生成成功")

        # 挂载碰撞传感器
        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(on_collision)
        actor_list.append(collision_sensor)

        # 生成背景车辆
        traffic_count = random.randint(10, 15)
        spawned_traffic = 0
        for _ in range(traffic_count):
            traffic_bp = random.choice(blueprint_library.filter('vehicle.*'))
            traffic_spawn = random.choice(map.get_spawn_points())
            traffic_vehicle = world.try_spawn_actor(traffic_bp, traffic_spawn)
            if traffic_vehicle:
                traffic_vehicle.set_autopilot(True)
                actor_list.append(traffic_vehicle)
                spawned_traffic += 1
        print(f"生成背景车辆：{spawned_traffic}辆")

        # 生成前向摄像头
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CONFIG["CAMERA_WIDTH"]))
        camera_bp.set_attribute("image_size_y", str(CONFIG["CAMERA_HEIGHT"]))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # 摄像头回调（线程安全）
        def image_callback(image):
            processed_img = process_image(image)
            with g_thread_lock:
                image_surface[0] = processed_img

        camera.listen(image_callback)

        # 初始化Pygame
        display = init_pygame(CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"])
        clock = pygame.time.Clock()

        # 第三人称视角更新
        spectator = world.get_spectator()

        def update_spectator():
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + transform.get_forward_vector() * -10 + carla.Location(z=8),
                carla.Rotation(pitch=-15, yaw=transform.rotation.yaw, roll=0)
            ))

        # 主循环
        running = True
        while running:
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                    break
            if not running:
                break

            update_spectator()
            control = carla.VehicleControl()

            # 安全读取车辆状态
            try:
                current_speed = get_speed(vehicle)
                vehicle_transform = vehicle.get_transform()
            except:
                print("警告：无法读取车辆状态，跳过本次循环")
                time.sleep(0.1)
                continue

            # ========== 碰撞重置逻辑（线程安全） ==========
            reset_flag = False
            with g_thread_lock:
                if need_vehicle_reset:
                    reset_flag = True
                    need_vehicle_reset = False

            if reset_flag:
                print("【碰撞保护】正在执行紧急重置...")
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
                vehicle.apply_control(control)
                time.sleep(0.5)

                new_spawn_point = random.choice(map.get_spawn_points())
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                time.sleep(0.1)
                vehicle.set_transform(new_spawn_point)
                print(f"【碰撞保护】车辆已重置到新位置")
                continue

            # ========== V5.0 核心修复：红绿灯通行全逻辑 ==========
            # 1. 获取停止线精准信息
            distance_to_stop, is_in_junction, traffic_light = get_stop_line_info(vehicle, map)
            # 2. 获取红绿灯状态（兜底：车辆当前触发区的红绿灯）
            if not traffic_light:
                traffic_light = vehicle.get_traffic_light()
            light_state = traffic_light.get_state() if traffic_light else carla.TrafficLightState.Green
            should_stop = False
            target_speed = CONFIG["CRUISE_SPEED"]

            # 3. 路口内不处理红绿灯（已过停止线，禁止停车）
            if not is_in_junction and traffic_light is not None:
                required_brake_dist = calc_required_brake_distance(current_speed)

                # 红灯逻辑：必须停车
                if light_state == carla.TrafficLightState.Red:
                    print(f"【红绿灯】红灯 | 到停止线：{distance_to_stop:.1f}m | 所需刹车距离：{required_brake_dist:.1f}m")
                    # 距离小于刹车距离，必须停车
                    if distance_to_stop <= required_brake_dist:
                        should_stop = True
                    # 距离大于刹车距离，提前减速到路口速度
                    elif distance_to_stop < 50:
                        target_speed = CONFIG["INTERSECTION_SPEED"]

                # 黄灯逻辑：减速准备停车，能安全停就停，过线就走
                elif light_state == carla.TrafficLightState.Yellow:
                    print(f"【红绿灯】黄灯 | 到停止线：{distance_to_stop:.1f}m")
                    if distance_to_stop <= CONFIG["YELLOW_SLOWDOWN_DISTANCE"] and distance_to_stop > CONFIG[
                        "MIN_STOP_DISTANCE"]:
                        target_speed = CONFIG["INTERSECTION_SPEED"]
                        # 能安全刹停就停车
                        if distance_to_stop <= required_brake_dist:
                            should_stop = True

            # 4. 停车控制逻辑（梯度刹车，避免急刹）
            if should_stop:
                # 完全刹停
                if distance_to_stop < CONFIG["MIN_STOP_DISTANCE"] or current_speed < 5:
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.steer = 0.0
                else:
                    # 梯度刹车，距离越近刹车力度越大
                    brake_strength = min(1.0, required_brake_dist / max(distance_to_stop, 1e-5))
                    control.throttle = 0.0
                    control.brake = brake_strength
                    control.steer = 0.0

            # 5. 正常行驶逻辑
            else:
                # 路径跟随
                waypoint = map.get_waypoint(vehicle_transform.location, project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
                next_waypoints = waypoint.next(2.0)
                if next_waypoints:
                    # 优先选同车道直行
                    next_waypoint = next_waypoints[0]
                    for wp in next_waypoints:
                        if wp.lane_id == waypoint.lane_id:
                            next_waypoint = wp
                            break
                    control.steer = get_steer(vehicle_transform, next_waypoint.transform)

                # 路口前提前减速
                if distance_to_stop < 50 and not is_in_junction:
                    target_speed = CONFIG["INTERSECTION_SPEED"]

                # 速度控制
                if current_speed < target_speed:
                    control.throttle = 0.5
                    control.brake = 0.0
                else:
                    control.throttle = 0.2
                    control.brake = 0.0

            # 应用车辆控制
            vehicle.apply_control(control)

            # 画面渲染（线程安全）
            current_frame = None
            with g_thread_lock:
                if image_surface[0] is not None:
                    current_frame = image_surface[0].copy()

            if current_frame is not None:
                surface = pygame.image.frombuffer(current_frame.tobytes(),
                                                  (CONFIG["CAMERA_WIDTH"], CONFIG["CAMERA_HEIGHT"]), "RGB")
                display.blit(surface, (0, 0))
                pygame.display.flip()

            clock.tick(30)

    except Exception as e:
        print(f"发生严重错误: {e}")
    finally:
        print("正在安全清理资源...")
        # 先停止所有传感器
        for actor in actor_list:
            if actor and 'sensor' in actor.type_id:
                try:
                    actor.stop()
                except:
                    pass
        time.sleep(0.5)
        # 销毁所有对象
        for actor in actor_list:
            if actor:
                try:
                    actor.destroy()
                except:
                    pass
        pygame.quit()
        print("程序结束")


if __name__ == "__main__":
    main()
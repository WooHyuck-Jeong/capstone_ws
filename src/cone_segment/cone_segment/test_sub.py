"""
test_sub.py
===========
꼬깔 검출 토픽 모니터링용 구독 노드

Subscribed Topics:
  /cone_pairs   (std_msgs/String)
  /cone_bearing (std_msgs/Float32MultiArray)

Usage:
  ros2 run cone_segment test_sub
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray


class ConeTestSub(Node):

    def __init__(self):
        super().__init__('cone_test_sub')

        self.sub_pairs = self.create_subscription(
            String,
            '/cone_pairs',
            self.cb_pairs,
            10
        )

        self.sub_bearing = self.create_subscription(
            Float32MultiArray,
            '/cone_bearing',
            self.cb_bearing,
            10
        )

        self.get_logger().info('ConeTestSub started')
        self.get_logger().info('Subscribing: /cone_pairs, /cone_bearing')

    def cb_pairs(self, msg: String):
        try:
            data = json.loads(msg.data)
            pairs = data.get('pairs', [])

            self.get_logger().info('='*50)
            self.get_logger().info(f'Pairs: {len(pairs)}')

            for p in pairs:
                self.get_logger().info(
                    f"  Pair{p['pair_index']} | "
                    f"R{p['red']['index']}({p['red']['cx']:.0f},{p['red']['cy']:.0f}) "
                    f"<-> "
                    f"B{p['blue']['index']}({p['blue']['cx']:.0f},{p['blue']['cy']:.0f}) "
                    f"| bearing={p['bearing_deg']:+.2f}deg"
                )
        except Exception as e:
            self.get_logger().error(f'Parse error: {e}')

    def cb_bearing(self, msg: Float32MultiArray):
        bearings = msg.data
        if not bearings:
            return
        bearing_str = ', '.join([f'{b:+.2f}deg' for b in bearings])
        self.get_logger().info(f'Bearings: [{bearing_str}]')


def main(args=None):
    rclpy.init(args=args)
    node = ConeTestSub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
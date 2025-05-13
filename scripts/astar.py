import heapq


class Node:
    """
    A* 算法中的节点类
    """

    def __init__(self, position, parent=None):
        self.position = position  # 节点在网格中的坐标 (row, col)
        self.parent = parent  # 父节点

        self.g = 0  # 从起点到当前节点的实际代价
        self.h = 0  # 从当前节点到终点的启发式估计代价
        self.f = 0  # 总评估代价 f = g + h

    def __eq__(self, other):
        # 用于判断两个节点是否相同（位置相同即视为相同）
        return self.position == other.position

    def __lt__(self, other):
        # 用于优先队列的比较，优先选择 f 值小的节点
        return self.f < other.f

    def __hash__(self):
        # 为了能将节点放入集合 (set) 中，需要实现 hash 方法
        return hash(self.position)


def heuristic(node_pos, goal_pos, method="manhattan"):
    """
    启发式函数，计算从当前节点到目标节点的估计代价

    参数:
    node_pos (tuple): 当前节点的坐标 (row, col)
    goal_pos (tuple): 目标节点的坐标 (row, col)
    method (str): 计算启发式代价的方法，可选 'manhattan' 或 'euclidean'

    返回:
    float: 估计的启发式代价
    """
    dx = abs(node_pos[0] - goal_pos[0])
    dy = abs(node_pos[1] - goal_pos[1])

    if method == "manhattan":
        # 曼哈顿距离：适用于只能在四个方向（上、下、左、右）移动的情况
        return dx + dy
    elif method == "euclidean":
        # 欧几里得距离：适用于可以在任意方向移动的情况
        import math

        return math.sqrt(dx**2 + dy**2)
    else:
        raise ValueError("未知的启发式方法。请选择 'manhattan' 或 'euclidean'")


def astar_search(
    grid, start_pos, goal_pos, heuristic_method="manhattan", allow_diagonal=False
):
    """
    A* 寻路算法实现

    参数:
    grid (list of list of int): 代表地图的二维网格。0 表示可通过，1 表示障碍物。
    start_pos (tuple): 起点坐标 (row, col)
    goal_pos (tuple): 终点坐标 (row, col)
    heuristic_method (str): 启发式函数计算方法
    allow_diagonal (bool): 是否允许对角线移动

    返回:
    list of tuples: 从起点到终点的路径坐标列表，如果找不到路径则返回 None
    """

    # 创建起点和终点节点
    start_node = Node(start_pos)
    goal_node = Node(goal_pos)

    # 初始化开放列表 (open_list) 和关闭列表 (closed_list)
    # open_list: 存放待考察的节点，使用优先队列（最小堆）实现，f 值最小的节点优先
    # closed_list: 存放已考察过的节点，使用集合 (set) 实现，方便快速查找
    open_list = []
    closed_list = set()

    # 将起点加入开放列表
    heapq.heappush(open_list, start_node)

    # 定义可能的移动方向
    # (row_change, col_change, cost)
    # 默认只允许上下左右移动，代价为 1
    moves = [(0, -1, 1), (0, 1, 1), (-1, 0, 1), (1, 0, 1)]  # 左  # 右  # 上  # 下
    if allow_diagonal:
        # 如果允许对角线移动，代价通常设为 sqrt(2) ≈ 1.414
        # 为了简化，这里也设为 1.4，或者可以根据实际情况调整
        # 注意：如果对角线移动的代价与直线移动代价不同，启发式函数的选择需要更小心
        # 例如，如果使用曼哈顿距离，但允许对角线移动，可能会导致非最优路径
        # 此时，对角线距离 (Chebyshev distance) 或欧几里得距离可能更合适
        moves.extend(
            [
                (-1, -1, 1.414),  # 左上
                (-1, 1, 1.414),  # 右上
                (1, -1, 1.414),  # 左下
                (1, 1, 1.414),  # 右下
            ]
        )

    grid_height = len(grid)
    grid_width = len(grid[0])

    # 主循环，直到开放列表为空或找到终点
    while open_list:
        # 从开放列表中取出 f 值最小的节点作为当前节点
        current_node = heapq.heappop(open_list)

        # 如果当前节点是终点，则已找到路径
        if current_node == goal_node:
            path = []
            temp = current_node
            while temp is not None:
                path.append(temp.position)
                temp = temp.parent
            return path[::-1]  # 返回反转后的路径（从起点到终点）

        # 将当前节点加入关闭列表
        closed_list.add(current_node)

        # 遍历当前节点的邻居节点
        for move_dr, move_dc, move_cost in moves:
            neighbor_pos = (
                current_node.position[0] + move_dr,
                current_node.position[1] + move_dc,
            )

            # 检查邻居节点是否在网格范围内
            if not (
                0 <= neighbor_pos[0] < grid_height and 0 <= neighbor_pos[1] < grid_width
            ):
                continue

            # 检查邻居节点是否是障碍物
            if grid[neighbor_pos[0]][neighbor_pos[1]] == 1:  # 假设 1 代表障碍物
                continue

            # 创建邻居节点
            neighbor_node = Node(neighbor_pos, current_node)

            # 如果邻居节点已在关闭列表中，则忽略
            if neighbor_node in closed_list:
                continue

            # 计算邻居节点的 g, h, f 值
            neighbor_node.g = current_node.g + move_cost  # 从起点到邻居的代价
            neighbor_node.h = heuristic(
                neighbor_node.position, goal_node.position, heuristic_method
            )
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            # 检查邻居节点是否已在开放列表中
            # 如果在，并且新的路径代价更低，则更新它
            # 如果不在，则加入开放列表
            found_in_open = False
            for i, open_node in enumerate(open_list):
                if open_node == neighbor_node:
                    found_in_open = True
                    if neighbor_node.g < open_node.g:  # 如果新的路径更优
                        open_list[i] = (
                            neighbor_node  # 更新节点（在 heapq 中直接替换可能不保证堆性质，更安全的做法是移除旧的，添加新的，或者标记旧的为无效）
                        )
                        # 为了简化，这里直接替换。对于性能要求极高的场景，可能需要更复杂的处理。
                        # Python 的 heapq 不直接支持 decrease-key 操作，通常的做法是添加新的，让旧的留在堆中，取出时判断是否已处理。
                        # 或者，如果节点可变且比较基于 f 值，直接修改 f 值后调用 heapq.heapify(open_list) 或 heapq.heappushpop/heapreplace。
                        # 这里我们假设 Node 的 __lt__ 是基于 f 值的，所以如果 g 变小导致 f 变小，
                        # 重新插入或调整堆是必要的。
                        # 一个更健壮（但可能稍慢）的方式是：
                        # open_list.pop(i) # 移除旧的（这会破坏堆结构，不推荐）
                        # heapq.heappush(open_list, neighbor_node)
                        # 或者，简单地再次推入，让 pop 时处理重复（如果 Node 的 __eq__ 只比较 position）
                        # 这里我们采用更简单的方式，如果发现性能瓶颈，可以优化这部分。
                        # 实际上，更常见的做法是，如果允许重复添加，那么在从堆中取出节点时，
                        # 如果该节点（基于位置）已经在 closed_list 中，则跳过。
                        # 但我们这里的 closed_list 检查在循环开始时。
                        # 更好的处理方式是，如果 open_list 中已存在且新路径更优，则更新其 g 值和 parent，然后重新调整堆。
                        # Python 的 heapq 不直接支持 decrease-key，所以通常是添加一个新项，旧项会被忽略（因为 f 值更高）。
                        # 或者，如果节点是可变的，并且比较是基于 f 值，则修改 f 值并重新堆化。
                        # 为了简单起见，我们这里假设如果找到了，就更新并重新排序（虽然 heapq.heappush 会处理排序）。
                        # 实际上，如果 neighbor_node.g < open_node.g，那么 open_node 的 parent 和 g 值也应该更新。
                        open_list[i].g = neighbor_node.g
                        open_list[i].parent = current_node  # 更新父节点
                        open_list[i].f = open_list[i].g + open_list[i].h  # 重新计算 f
                        heapq.heapify(open_list)  # 重新构建堆
                    break  # 已找到，无需继续在 open_list 中搜索

            if not found_in_open:
                heapq.heappush(open_list, neighbor_node)

    return None  # 如果开放列表为空仍未找到终点，则说明没有路径


# 示例用法
if __name__ == "__main__":
    # 定义网格 (0: 可通行, 1: 障碍物)
    grid = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 一条通路
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (7, 6)

    # 从 grid 随机找到两个点作为 start 和 goal
    import random

    start = (random.randint(0, len(grid) - 1), random.randint(0, len(grid[0]) - 1))
    goal = (random.randint(0, len(grid) - 1), random.randint(0, len(grid[0]) - 1))
    while grid[start[0]][start[1]] == 1:
        start = (random.randint(0, len(grid) - 1), random.randint(0, len(grid[0]) - 1))
    while grid[goal[0]][goal[1]] == 1 or goal == start:
        goal = (random.randint(0, len(grid) - 1), random.randint(0, len(grid[0]) - 1))

    print(f"起点: {start}, 终点: {goal}")
    print("使用曼哈顿距离，不允许对角线移动:")
    path = astar_search(
        grid, start, goal, heuristic_method="manhattan", allow_diagonal=False
    )

    if path:
        print("找到路径:")
        print(path)
        for r in range(len(grid)):
            row_str = ""
            for c in range(len(grid[0])):
                if (r, c) == start:
                    row_str += "S "
                elif (r, c) == goal:
                    row_str += "G "
                elif (r, c) in path:
                    row_str += "* "
                elif grid[r][c] == 1:
                    row_str += "# "  # 障碍物
                else:
                    row_str += ". "  # 可通行
            print(row_str)
    else:
        print("未找到路径")

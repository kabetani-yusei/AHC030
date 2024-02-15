import sys
import math
import numpy as np

'''
num: ブロックの数
coordinate: 長方形とした時の[x,y]の最大値
area: 長方形とした時の面積
blocks_list: 各ブロックの位置
'''

class Block:
    def __init__(self, num: int, coordinate: 'list[int]', area: int, blocks_list: 'list[list[int]]'):
        self.num = num
        self.coordinate = coordinate
        self.area = area
        self.blocks_list = blocks_list
        
    def __eq__(self, other):
        if isinstance(other, Block):
            return (
                self.num == other.num and
                self.coordinate == other.coordinate and
                self.area == other.area and
                self.blocks_list == other.blocks_list
            )
        return False
    
    def duplicate(self, n: int):
        self.duplication = n
        
    
     
class Mapping:
    
    def __init__(self, n: int, m: int):
        self.MAX_RECURSION_DEPTH = 1000 # この回数を超えた場合は終了する
        self.roop_count = 0
        
        self.n = n
        self.m = m
        self.ans_cand = []
        self.block_look_order = []

        self.fixed_blocks_list = []
        self.fixed_blocks = [[] for _ in range(m)]
        
    def input_list(self, val_sum_ac:int, blocks: 'list[Block]') -> None:
        self.val_sum_ac = val_sum_ac
        self.blocks = blocks
        
    def predict(self, val_sum:int, original_map: 'list[list[int]]') -> 'list[list[list[int]]]':
        self.ans_cand = []
        self.val_sum = val_sum
        self.block_look_order = [x for x in self.fixed_blocks_list]
        self.roop_count = 0
        for i in range(self.m):
            if i not in self.block_look_order:
                self.block_look_order.append(i)
        map = [x[:] for x in original_map]
        self.dfs(0, 0, [], map)
        if self.roop_count > self.MAX_RECURSION_DEPTH or self.ans_cand == []:
            return []
        # 各要素をタプルに変換してセットに格納
        unique_set = set(tuple(tuple(inner) for inner in outer) for outer in self.ans_cand)
        unique_list = [list(list(inner) for inner in outer) for outer in unique_set]
        return unique_list
    
    
    def dfs(self, index: int, val_now:int, ans_list: 'list[tuple[int]]', map: 'list[list[int]]') -> None:
        if self.roop_count > self.MAX_RECURSION_DEPTH:
            return
        if index == self.m:
            for i in range(self.n):
                for j in range(self.n):
                    if map[i][j] >= 1:
                        return
            self.ans_cand.append([list(x) for x in ans_list])
            return

        idx = self.block_look_order[index]
        if idx in self.fixed_blocks_list:
            temp_map = [x[:] for x in map]
            each_block_list_temp = []
            for cc in self.blocks[idx].blocks_list:
                ii = cc[0] + self.fixed_blocks[idx][0]
                jj = cc[1] + self.fixed_blocks[idx][1]
                each_block_list_temp.append((ii,jj))
                if temp_map[ii][jj] >= 1:
                    temp_map[ii][jj] -= 1
                else:
                    val_now += 1
            self.dfs(index + 1, val_now, ans_list + each_block_list_temp, temp_map)
        else:
            looking_block = self.blocks[idx]
            looking_block_coordinate = looking_block.coordinate
            for i in range(self.n - looking_block_coordinate[0]):
                for j in range(self.n - looking_block_coordinate[1]):
                    each_block_list_temp = []
                    flag = True
                    for cc in looking_block.blocks_list:
                        if map[cc[0] + i][cc[1] + j] == 0:
                            flag = False
                            break
                        each_block_list_temp.append((cc[0] + i, cc[1] + j))

                    if flag:
                        val_temp = 0
                        temp_map = [x[:] for x in map]
                        for cc in looking_block.blocks_list:
                            if temp_map[cc[0] + i][cc[1] + j] >= 1:
                                temp_map[cc[0] + i][cc[1] + j] -= 1
                            else:
                                val_temp += 1
                        self.roop_count += 1
                        if self.roop_count > self.MAX_RECURSION_DEPTH:
                            return
                        if (val_now+val_temp) + self.val_sum <= self.val_sum_ac:
                            self.dfs(index + 1, (val_now+val_temp), ans_list + each_block_list_temp, temp_map)


    def first_map(self, map: 'list[list[int]]') -> 'list[list[int]]':
        for looking_block in self.blocks:
            looking_block_coordinate = looking_block.coordinate
            for i in range(self.n - looking_block_coordinate[0]):
                for j in range(self.n - looking_block_coordinate[1]):
                    for cc in looking_block.blocks_list:
                        map[cc[0] + i][cc[1] + j] = -1
        return map
    
    def reset(self, map: 'list[list[int]]') -> 'tuple[int, list[list[int]]]':
        self.block_look_order = [x for x in self.fixed_blocks_list]
        for i in range(self.m):
            if i not in self.block_look_order:
                self.block_look_order.append(i)
        temp_map = [x[:] for x in map]
        return (0, temp_map)
    def reflection(self, map: 'list[list[int]]') -> 'tuple[list[list[int]], int, list[list[int, list[int]]]]':
        idx, temp_map = self.reset(map)
        while(idx < self.m):
            block_idx = self.block_look_order[idx]
            if block_idx in self.fixed_blocks_list:
                for cc in self.blocks[block_idx].blocks_list:
                    if temp_map[cc[0] + self.fixed_blocks[block_idx][0]][cc[1] + self.fixed_blocks[block_idx][1]] >= 1:
                        temp_map[cc[0] + self.fixed_blocks[block_idx][0]][cc[1] + self.fixed_blocks[block_idx][1]] -= 1
            
            else:
                fit_count = 0
                block_coordinate = self.blocks[block_idx].coordinate
                for i in range(self.n - block_coordinate[0]):
                    for j in range(self.n - block_coordinate[1]):
                        flag = True
                        for cc in self.blocks[block_idx].blocks_list:
                            if temp_map[cc[0] + i][cc[1] + j] == 0:
                                flag = False
                                break
                        if flag:
                            fit_count += 1
                            self.fixed_blocks[block_idx] = [i, j]
                if fit_count <= self.blocks[block_idx].duplication:
                    self.fixed_blocks_list.append(block_idx)
                    idx, temp_map = self.reset(map)
                    continue
            idx += 1
        
        #埋めていく & エントロピーによる優先順位    
        temp_map_zero = [[0]*self.n for _ in range(self.n)]
        temp_map_h_sum = [[0]*self.n for _ in range(self.n)]
        idx, temp_map = self.reset(map)
        for idx in range(self.m):
            block_idx = self.block_look_order[idx]
            if block_idx in self.fixed_blocks_list:
                for cc in self.blocks[block_idx].blocks_list:
                    x = cc[0] + self.fixed_blocks[block_idx][0]
                    y = cc[1] + self.fixed_blocks[block_idx][1]
                    temp_map_zero[x][y] += 1
                    if temp_map[x][y] >= 1:
                        temp_map[x][y] -= 1
            
            else:
                temp_map_h = [[0]*self.n for _ in range(self.n)]
                block_coordinate = self.blocks[block_idx].coordinate
                fit_count = 0
                for i in range(self.n - block_coordinate[0]):
                    for j in range(self.n - block_coordinate[1]):
                        flag = True
                        for cc in self.blocks[block_idx].blocks_list:
                            if temp_map[cc[0] + i][cc[1] + j] == 0:
                                flag = False
                                break
                        if flag:
                            fit_count += 1
                            self.fixed_blocks[block_idx] = [i, j]
                            for cc in self.blocks[block_idx].blocks_list:
                                temp_map_zero[cc[0] + i][cc[1] + j] += 1
                                temp_map_h[cc[0] + i][cc[1] + j] += 1
                for i in range(self.n):
                    for j in range(self.n):
                        if temp_map_h[i][j] > 0:
                            prob = round(temp_map_h[i][j] / fit_count, 5)
                            temp_map_h_sum[i][j] = round(temp_map_h_sum[i][j] - prob * math.log2(prob), 5)
            
        for i in range(self.n):
            for j in range(self.n):
                if temp_map_zero[i][j] == 0 and map[i][j] == -1:
                    map[i][j] = 0       
        cell_cand = []
        for i in range(self.n):
            for j in range(self.n):
                if map[i][j] == -1:
                    cell_cand.append((temp_map_h_sum[i][j], [i, j]))

        return (map, cell_cand)
    
    
    def cell_information_ans(self, map: 'list[list[int]]', ans_cand: 'list[list[list[int]]]') -> 'list[list[float, list[int]]]':
        temp_map_h_sum = [[0]*self.n for _ in range(self.n)]
        temp_map_h = [[{} for _ in range(self.n)] for _ in range(self.n)]

        fit_count = len(ans_cand)
        for cand in ans_cand:
            temp_map_cand = [[0]*self.n for _ in range(self.n)]
            for cc in cand:
                temp_map_cand[cc[0]][cc[1]] += 1
            for i in range(self.n):
                for j in range(self.n):
                    if temp_map_cand[i][j] not in temp_map_h[i][j]:
                        temp_map_h[i][j][temp_map_cand[i][j]] = 1
                    else:
                        temp_map_h[i][j][temp_map_cand[i][j]] += 1
        
        cell_cand = []
        for i in range(self.n):
            for j in range(self.n):
                if map[i][j] == -1:
                    for value in temp_map_h[i][j].values():
                        prob = round(value / fit_count, 5)
                        temp_map_h_sum[i][j] = round(temp_map_h_sum[i][j] - prob * math.log2(prob), 5)
                    cell_cand.append((temp_map_h_sum[i][j], [i, j]))
        return cell_cand
        
class Judge:

    def __init__(self, n: int, m: int, e: float):
        self.n = n
        self.m = m
        self.e = e

    def read_blocks(self) -> 'tuple[int, list[Block]]':
        blocks = []
        val_sum = 0
        for _ in range(self.m):
            t = list(map(int, input().split()))
            blocks_list = [t[i:i+2] for i in range(1, len(t), 2)]
            coordinate = [max([x[0] for x in blocks_list]), max([x[1] for x in blocks_list])]
            area = coordinate[0] * coordinate[1]
            blocks.append(Block(t[0], coordinate, area, blocks_list))
            val_sum += t[0]
        blocks.sort(key=lambda x: x.area, reverse=True)
        for i in range(self.m):
            flag = 0
            for j in range(self.m):
                if blocks[i] == blocks[j]:
                    flag += 1
            blocks[i].duplicate(flag)
        return (val_sum, blocks)
    
    def output_query(self, use_choice: str, use_area: int, use_place: str) -> None:
        query = f"{use_choice} {use_area} {use_place}"
        print(query, flush=True)
        
    def read_response(self) -> int:
        return int(input())


class Visualizer():
    def __init__(self, n: int, m: int, e: float):
        print("# Visualizer mode")
        self.n = n
        self.m = m
        self.e = e
        self.val_sum = 0
        self.blocks = []
        self.ans_map = [[0] * n for _ in range(n)]
        self.ans_list = []
        self.response = 0
        self.cost = 0.0
        
        #入力の読み込み
        #各ブロックの形
        for _ in range(self.m):
            t = list(map(int, input().split()))
            blocks_list = [t[i:i+2] for i in range(1, len(t), 2)]
            coordinate = [max([x[0] for x in blocks_list]), max([x[1] for x in blocks_list])]
            area = coordinate[0] * coordinate[1]
            self.blocks.append(Block(t[0], coordinate, area, blocks_list))
            self.val_sum += t[0]
        self.blocks.sort(key=lambda x: x.area, reverse=True)
        for i in range(self.m):
            flag = 0
            for j in range(self.m):
                if self.blocks[i] == self.blocks[j]:
                    flag += 1
            self.blocks[i].duplicate(flag)
        
        #各ブロックの位置
        for _ in range(self.m):
            _, _ = map(int, input().split())
        
        #マップの情報
        for i in range(self.n):
            line_input = list(map(int, input().split()))
            for j in range(self.n):
                self.ans_map[i][j] = line_input[j]
                if line_input[j] >= 1:
                    self.ans_list.append([i, j])
        self.ans_list.sort()
                
        #誤差の情報
        for _ in range(2 * self.n ** 2):
            _ = input()
        
        
    def read_blocks(self) -> 'tuple[int, list[Block]]':
        return (self.val_sum, self.blocks)
    
    def output_query(self, use_choice: str, use_area: int, use_place: str) -> None:
        if use_choice == 'a':
            self.cost += 1
        else:
            if use_area == 1:
                self.cost += 1
            else:
                self.cost = round(self.cost + 1 / math.sqrt(use_area), 5)
                
        query = f"{use_choice} {use_area} {use_place}"
        print(query, flush=True)
        self.response = 0
        if use_choice == 'a':
            temp = list(map(int, use_place.split()))
            temp_blocks = [temp[i:i+2] for i in range(0, len(temp), 2)]
            self.response = 0
            if sorted(temp_blocks) == self.ans_list:
                self.response = 1          
        elif use_choice == 'q' and use_area == 1:
            temp = list(map(int, use_place.split()))
            self.response = self.ans_map[temp[0]][temp[1]]          
        elif use_choice == 'q' and use_area >= 2:
            temp = list(map(int, use_place.split()))
            area_list = [temp[i:i+2] for i in range(0, len(temp), 2)]
            area_sum = sum([self.ans_map[x[0]][x[1]] for x in area_list])
            u = (use_area - area_sum) * self.e + area_sum * (1 - self.e)
            v = use_area * (1 - self.e) * self.e
            # 平均が u で分散が v の正規分布から一つの要素を抽出
            sample = np.random.normal(u, np.sqrt(v))
            self.response = max(0, round(sample))
            
    def read_response(self) -> int:
        return self.response

    def comment(self, i:int, j:int, val:int) -> None:
        if self.ans_map[i][j] != val:
            print(f"#c {i} {j} red")
        elif val == 0:
            print(f"#c {i} {j} #999999")
        elif val == 1:
            print(f"#c {i} {j} #666666")
        elif val == 2:
            print(f"#c {i} {j} #333333")
        elif val == 3:
            print(f"#c {i} {j} #112233")
        elif val == 4:
            print(f"#c {i} {j} #332211")
        elif val >= 5:
            print(f"#c {i} {j} #0000FF")
              
           
        
          
class Solver:

    def __init__(self, n: int, m: int, e: float, mode: int):
        print(f"n:{n}, m:{m}, e:{e}", file=sys.stderr)
        self.n = n
        self.m = m
        self.e = e
        self.mode = mode
        
        self.turn = 0
        self.clear_flag = False
        self.map = [[0] * self.n for _ in range(self.n)]
        
        self.ans_cand = []
        self.used_question = {}
        self.val_sum = 0
        self.val_sum_ac = 0
        
        self.mapping_class = Mapping(self.n, self.m)
        
        if mode == 0:
            self.judge = Judge(n, m, e)
        else:
            self.judge = Visualizer(n, m, e)

    
    def solve(self) -> int:
        self.val_sum_ac, self.blocks = self.judge.read_blocks()
        self.mapping_class.input_list(self.val_sum_ac, self.blocks)
        self.map = self.mapping_class.first_map(self.map)

        for _ in range(2 * self.n ** 2):
            use_choice, use_area, use_place = self.select_action()
            if use_choice == 'q' and use_area == 1:
                self.judge.output_query(use_choice, use_area, self.flatten_list([use_place]))
                res = self.judge.read_response()
                self.reflection_response(use_place, res)
                
            elif use_choice == 'q' and use_area >= 2:
                print("作成中です", file=sys.stderr)
                
            elif use_choice == 'a':
                ans_query_string = self.flatten_list(use_place)
                #過去に同じ質問をしたかどうかの判定
                if self.used_question.get(ans_query_string) is None:
                    self.used_question[ans_query_string] = 1
                    self.judge.output_query(use_choice, use_area, ans_query_string)
                    res = self.judge.read_response()
                else:
                    res = 0
            
            if self.while_end_process(use_choice, res):
                break                   
        
        return self.solve_end_process()
    
    def select_action(self) -> 'tuple[str, int, list[list[int]]]':
        if self.val_sum == self.val_sum_ac:
            ans_list = []
            for i in range(self.n):
                for j in range(self.n):
                    if self.map[i][j] >= 1:
                        ans_list.append([i, j])
            return ('a', len(ans_list), ans_list)
        
        self.map, cell_cand = self.mapping_class.reflection(self.map)
        self.now_map()
        if self.ans_cand == []:
            self.ans_cand = self.mapping_class.predict(self.val_sum, self.map)
        while(self.ans_cand != [] and len(self.ans_cand) <= 3):
            ans_list = self.ans_cand.pop()
            ans_list = set(tuple(inner) for inner in ans_list)
            ans_list = [list(inner) for inner in ans_list]
            return ('a', len(ans_list), ans_list)
        
        if self.ans_cand != []:
            cell_cand_ans = self.mapping_class.cell_information_ans(self.map, self.ans_cand)
            cell_cand_ans_sorted = sorted(cell_cand_ans, key=lambda x: x[0], reverse=True)
            self.ans_cand = []
            return ('q', 1, cell_cand_ans_sorted[0][1])
        
        cell_cand_sorted = sorted(cell_cand, key=lambda x: x[0], reverse=True)
        return ('q', 1, cell_cand_sorted[0][1])

                       
    def reflection_response(self, use_place: 'list[int]', response: int) -> None:
        if self.map[use_place[0]][use_place[1]] == -1:
            self.map[use_place[0]][use_place[1]] = response
            self.val_sum += response
        
    def flatten_list(self, nested_list):
        nested_list.sort(key=lambda x: [x[0],x[1]])
        flattened_list = [item for sublist in nested_list for item in sublist]
        return ' '.join(map(str, flattened_list))
    
    def now_map(self) -> None:
        if self.mode == 1:
            for i in range(self.n):
                for j in range(self.n):
                    if self.map[i][j] != -1:
                        self.judge.comment(i,j,self.map[i][j])
                    
    def while_end_process(self, use_choice: str, res: int) -> bool:
        self.turn += 1
        if use_choice == 'a' and res == 1:
            self.clear_flag = True
            return True
            
        if self.turn >= 2 * self.n ** 2:
            return True
        return False
        
    def solve_end_process(self) -> int:
        if self.mode == 0:
            return 0   
        else:
            if self.clear_flag == False:
                return 10 ** 6
            else:
                return self.judge.cost - 1.0          
                    
                    
def main():
    #コマンドライン引数がある場合はビジュアライザー用として処理する
    # mode  0: 通常, 1: ビジュアライザー用
    mode = 0
    if len(sys.argv) == 2:
        mode = 1
    n, m, e = input().split()
    n = int(n)
    m = int(m)
    e = float(e)
    solver = Solver(n, m, e, mode)
    cost = solver.solve()
    print(f"{cost}", file=sys.stderr)
    print(f"#total_cost:{cost}")


if __name__ == "__main__":
    main()

from dataclasses import dataclass
import sys
import math
import numpy as np
    
@dataclass
class Brock:
    blocks_list: 'list[list[int]]'
     
     
        
class Judge:

    def __init__(self, n: int, m: int, e: float):
        self.n = n
        self.m = m
        self.e = e

    def read_blocks(self) -> 'tuple[int, list[Brock]]':
        blocks = []
        value_sum = 0
        for _ in range(self.m):
            t = list(map(int, input().split()))
            blocks.append([Brock([t[i:i+2] for i in range(1, len(t), 2)])])
            value_sum += t[0]
        return (value_sum, blocks)
    
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
        self.blocks = []
        self.value_sum = 0
        self.ans_map = [[0] * n for _ in range(n)]
        self.ans_list = []
        self.response = 0
        
        #入力の読み込み
        #各ブロックの形
        for _ in range(self.m):
            t = list(map(int, input().split()))
            self.blocks.append([Brock([t[i:i+2] for i in range(1, len(t), 2)])])
            self.value_sum += t[0]
        
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
        
        
    def read_blocks(self) -> 'tuple[int, list[Brock]]':
        return (self.value_sum, self.blocks)
    
    def output_query(self, use_choice: str, use_area: int, use_place: str) -> None:
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

    def comment(self, message: str) -> None:
        print(f"# {message}")    
           
        
          
class Solver:

    def __init__(self, n: int, m: int, e: float, mode: int):
        print(f"n:{n}, m:{m}, e:{e}", file=sys.stderr)
        self.n = n
        self.m = m
        self.e = e
        self.value_sum = 0
        self.ac_value_sum = 0
        self.mode = mode
        if mode == 0:
            self.judge = Judge(n, m, e)
        else:
            self.judge = Visualizer(n, m, e)

    def flatten_list(self, nested_list):
        flattened_list = [item for sublist in nested_list for item in sublist]
        return ' '.join(map(str, flattened_list))
    
    def cost_change(self, cost: float, choice: str, area: int) -> int:
        if choice == 'a':
            cost += 1
        else:
            cost += round(1.0 / (math.sqrt(area)), 5)
        return cost
    
    def solve(self) -> int:
        self.turn = 0
        self.cost = 0.0
        self.clear_flag = False
        self.map = [[-1] * self.n for _ in range(self.n)]
        self.ac_value_sum, self.blocks = self.judge.read_blocks()
        
        for _ in range(2 * self.n ** 2):
            use_choice, use_area, use_place = self.select_action()
            self.judge.output_query(use_choice, use_area, self.flatten_list(use_place))
            res = self.judge.read_response()
            self.reflection_response(use_place, res)
            if use_choice == 'a' and res == 1:
                self.clear_flag = True
                break
            
            #得点計算用の処理
            if self.mode == 1:
                self.turn += 1
                self.cost = self.cost_change(self.cost, use_choice, use_area)
        
        return self.cost
    
    def select_action(self) -> 'tuple[str, int, list[list[int]]]':
        for i in range(self.n):
            for j in range(self.n):
                if self.map[i][j] == -1 and self.ac_value_sum > self.value_sum:
                    return ('q', 1, [[i, j]])
        
        ans_list = []
        for i in range(self.n):
            for j in range(self.n):
                if self.map[i][j] >= 1:
                    ans_list.append([i, j])
        return ('a', len(ans_list), ans_list)
                
    def reflection_response(self, use_place: 'list[list[int]]', response: int) -> None:
        self.map[use_place[0][0]][use_place[0][1]] = response
        self.value_sum += response
        

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

/// 终末地源石电路模块解谜 - 求解算法
/// 此文件包含分步记录框架和辅助工具，算法主体 (solve) 由你实现。
module EndfieldPuzzleSolver.Algorithm.Solver

open EndfieldPuzzleSolver.Recognition.Models

// ──────────────────────── 棋盘操作工具 ────────────────────────

/// 深拷贝二维棋盘
let cloneBoard (board: TileInfo[,]) : TileInfo[,] =
    let rows = Array2D.length1 board
    let cols = Array2D.length2 board
    Array2D.init rows cols (fun r c -> board.[r, c])

/// 在棋盘上放置一个元件（原地修改）
/// 返回被覆盖的原始格子列表（用于撤销）
let placeComponent
    (board: TileInfo[,])
    (comp: ComponentInfo)
    (compIdx: int)
    (row: int) (col: int)
    (rotation: int)
    : (int * int * TileInfo) list =

    let shape = comp.GetRotatedShape(rotation)
    let sr = shape.GetLength(0)
    let sc = shape.GetLength(1)
    let mutable undoList = []

    for dr in 0 .. sr - 1 do
        for dc in 0 .. sc - 1 do
            if shape.[dr, dc] then
                let r, c = row + dr, col + dc
                undoList <- (r, c, board.[r, c]) :: undoList
                board.[r, c] <- TileInfo.Placed(comp.ColorGroup, compIdx)

    undoList

/// 撤销放置（从 undoList 恢复原始格子）
let undoPlace (board: TileInfo[,]) (undoList: (int * int * TileInfo) list) =
    for (r, c, orig) in undoList do
        board.[r, c] <- orig

/// 检查元件能否放在指定位置（不越界、目标格子为 Empty）
let canPlace
    (board: TileInfo[,])
    (comp: ComponentInfo)
    (row: int) (col: int)
    (rotation: int)
    : bool =

    let shape = comp.GetRotatedShape(rotation)
    let sr = shape.GetLength(0)
    let sc = shape.GetLength(1)
    let boardRows = Array2D.length1 board
    let boardCols = Array2D.length2 board

    let mutable ok = true
    for dr in 0 .. sr - 1 do
        for dc in 0 .. sc - 1 do
            if shape.[dr, dc] then
                let r, c = row + dr, col + dc
                if r < 0 || r >= boardRows || c < 0 || c >= boardCols then
                    ok <- false
                elif board.[r, c].Type <> TileType.Empty then
                    ok <- false
    ok

// ──────────────────────── 需求检查工具 ────────────────────────

/// 统计棋盘上某行中各颜色组的格子数（包含 Lock 和 Placed）
let countRowColors (board: TileInfo[,]) (row: int) : Map<string, int> =
    let cols = Array2D.length2 board
    let mutable counts = Map.empty
    for c in 0 .. cols - 1 do
        let tile = board.[row, c]
        match tile.ColorGroup with
        | null -> ()
        | cg ->
            if tile.Type = TileType.Lock then
                counts <- counts |> Map.change cg (fun v -> Some((defaultArg v 0) + 1))
    counts

/// 统计棋盘上某列中各颜色组的格子数
let countColColors (board: TileInfo[,]) (col: int) : Map<string, int> =
    let rows = Array2D.length1 board
    let mutable counts = Map.empty
    for r in 0 .. rows - 1 do
        let tile = board.[r, col]
        match tile.ColorGroup with
        | null -> ()
        | cg ->
            if tile.Type = TileType.Lock then
                counts <- counts |> Map.change cg (fun v -> Some((defaultArg v 0) + 1))
    counts

/// 检查当前棋盘是否满足所有行列需求（用于最终验证）
let checkAllRequirements (puzzle: PuzzleData) (board: TileInfo[,]) : bool =
    let mutable ok = true

    // 检查每行
    for r in 0 .. puzzle.Rows - 1 do
        let actual = countRowColors board r
        for req in puzzle.RowRequirements.[r] do
            let actualCount = actual |> Map.tryFind req.ColorGroup |> Option.defaultValue 0
            if actualCount <> req.Count then
                ok <- false

    // 检查每列
    for c in 0 .. puzzle.Cols - 1 do
        let actual = countColColors board c
        for req in puzzle.ColumnRequirements.[c] do
            let actualCount = actual |> Map.tryFind req.ColorGroup |> Option.defaultValue 0
            if actualCount <> req.Count then
                ok <- false

    ok

// ──────────────────────── 分步记录器 ────────────────────────

/// 步骤记录器：在求解过程中记录每一步放置操作。
/// UI 层通过读取 Steps 实现逐步回放。
type StepRecorder(board: TileInfo[,]) =
    let steps = System.Collections.Generic.List<SolveStep>()

    /// 记录一步放置（拍摄当前棋盘快照）
    member _.Record(compIdx: int, row: int, col: int, rotation: int) =
        steps.Add(
            SolveStep(
                ComponentIndex = compIdx,
                Row = row,
                Col = col,
                Rotation = rotation,
                BoardSnapshot = cloneBoard board
            )
        )

    /// 撤销最后一步记录（回溯时调用）
    member _.UndoRecord() =
        if steps.Count > 0 then
            steps.RemoveAt(steps.Count - 1)

    /// 获取已记录的所有步骤
    member _.Steps = steps |> Seq.toList

// ──────────────────────── 求解入口 ────────────────────────

/// 对元件的某个旋转形状计算规范化哈希，用于去重
let private shapeKey (shape: bool[,]) =
    let rows = shape.GetLength(0)
    let cols = shape.GetLength(1)
    let sb = System.Text.StringBuilder()
    for r in 0 .. rows - 1 do
        for c in 0 .. cols - 1 do
            sb.Append(if shape.[r, c] then '1' else '0') |> ignore
        sb.Append('|') |> ignore
    sb.ToString()

/// 获取元件的所有不重复旋转角度列表
let private uniqueRotations (comp: ComponentInfo) =
    let mutable seen = Set.empty
    [ for rot in 0 .. 3 do
        let shape = comp.GetRotatedShape(rot)
        let key = shapeKey shape
        if not (Set.contains key seen) then
            seen <- Set.add key seen
            yield rot ]

/// 求解谜题，返回 SolveResult。
///
/// 算法：递归回溯 + 约束传播剪枝
/// 1. 按元件顺序逐个尝试放置
/// 2. 每次放置后检查受影响的行/列是否已超出颜色需求（剪枝）
/// 3. 所有元件放完后做最终需求验证
let solve (puzzle: PuzzleData) : SolveResult =

    let board = cloneBoard puzzle.Tiles
    let recorder = StepRecorder(board)
    let boardRows = puzzle.Rows
    let boardCols = puzzle.Cols

    // 预计算每行每列的颜色需求 Map<colorGroup, count>
    let rowReqs =
        Array.init boardRows (fun r ->
            puzzle.RowRequirements.[r]
            |> Array.fold (fun m (req: ColorRequirement) ->
                Map.add req.ColorGroup req.Count m) Map.empty)

    let colReqs =
        Array.init boardCols (fun c ->
            puzzle.ColumnRequirements.[c]
            |> Array.fold (fun m (req: ColorRequirement) ->
                Map.add req.ColorGroup req.Count m) Map.empty)

    // 预计算每个元件的不重复旋转列表
    let compRotations =
        puzzle.Components |> Array.map uniqueRotations

    /// 检查受影响的行列是否违反约束（放置后颜色数不超过需求）
    let checkAffectedConstraints (undoList: (int * int * TileInfo) list) =
        // 收集受影响的行号和列号
        let mutable affectedRows = Set.empty
        let mutable affectedCols = Set.empty
        for (r, c, _) in undoList do
            affectedRows <- Set.add r affectedRows
            affectedCols <- Set.add c affectedCols

        let mutable ok = true

        for r in affectedRows do
            if ok then
                let actual = countRowColors board r
                for kv in actual do
                    if ok then
                        let required = rowReqs.[r] |> Map.tryFind kv.Key |> Option.defaultValue 0
                        if kv.Value > required then
                            ok <- false

        for c in affectedCols do
            if ok then
                let actual = countColColors board c
                for kv in actual do
                    if ok then
                        let required = colReqs.[c] |> Map.tryFind kv.Key |> Option.defaultValue 0
                        if kv.Value > required then
                            ok <- false
        ok

    /// 递归回溯：尝试放置 remaining 列表中的所有元件
    let rec tryPlaceAll (remaining: int list) =
        match remaining with
        | [] ->
            // 所有元件已放置，最终验证全部行列需求
            checkAllRequirements puzzle board
        | compIdx :: rest ->
            let comp = puzzle.Components.[compIdx]
            let rotations = compRotations.[compIdx]
            let mutable solved = false

            for rotation in rotations do
                if not solved then
                    for r in 0 .. boardRows - 1 do
                        if not solved then
                            for c in 0 .. boardCols - 1 do
                                if not solved && canPlace board comp r c rotation then
                                    // 放置元件
                                    let undo = placeComponent board comp compIdx r c rotation
                                    // 剪枝：检查受影响行列的约束
                                    if checkAffectedConstraints undo then
                                        recorder.Record(compIdx, r, c, rotation)
                                        // 递归处理剩余元件
                                        if tryPlaceAll rest then
                                            solved <- true
                                        else
                                            // 回溯：撤销记录和放置
                                            recorder.UndoRecord()
                                            undoPlace board undo
                                    else
                                        // 约束违反，直接撤销
                                        undoPlace board undo
            solved

    let componentIndices = [ 0 .. puzzle.Components.Length - 1 ]

    if tryPlaceAll componentIndices then
        SolveResult.Solved(ResizeArray(recorder.Steps))
    else
        SolveResult.NoSolution(null)

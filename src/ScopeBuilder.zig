const std = @import("std");

const Ast = @import("Ast.zig");
const node = @import("node.zig");
const Code = @import("Code.zig");
const GeneralContext = @import("GeneralContext.zig");

const ScopeBuilder = @This();

ast: *Ast,
code: *Code,
arena: std.heap.ArenaAllocator,
scopes: std.SegmentedList(*node.Scope, 128) = .{},

pub fn init(ast: *Ast, code: *Code) ScopeBuilder {
    return .{
        .ast = ast,
        .code = code,
        .arena = ast.ctx.createLifetime(),
    };
}

pub fn deinit(b: *ScopeBuilder) void {
    b.arena.deinit();
}

// -- Manage scopes --

pub fn enterSourceFile(b: *ScopeBuilder, source_file: *node.SourceFile) void {
    source_file.scope = b.push();
}

pub fn exitSourceFile(b: *ScopeBuilder, source_file: *node.SourceFile) void {
    b.pop(source_file.scope.?);
}

pub fn enterSumType(b: *ScopeBuilder, sum_type: *node.SumType) void {
    sum_type.scope = b.push();
}

pub fn exitSumType(b: *ScopeBuilder, sum_type: *node.SumType) void {
    b.pop(sum_type.scope.?);
}

pub fn enterStructType(b: *ScopeBuilder, struct_type: *node.StructType) void {
    struct_type.scope = b.push();
}

pub fn exitStructType(b: *ScopeBuilder, struct_type: *node.StructType) void {
    b.pop(struct_type.scope.?);
}

pub fn enterFunDecl(b: *ScopeBuilder, fun_decl: *node.FunDecl) void {
    // TODO handle scoped ident: b.insert(fun_decl);
    fun_decl.scope = b.push();
}

pub fn exitFunDecl(b: *ScopeBuilder, fun_decl: *node.FunDecl) void {
    b.pop(fun_decl.scope.?);
}

pub fn enterCompStmt(b: *ScopeBuilder, comp_stmt: *node.CompStmt) void {
    comp_stmt.scope = b.push();
}

pub fn exitCompStmt(b: *ScopeBuilder, comp_stmt: *node.CompStmt) void {
    b.pop(comp_stmt.scope.?);
}

pub fn enterEnumType(b: *ScopeBuilder, enum_type: *node.EnumType) void {
    enum_type.scope = b.push();
}

pub fn exitEnumType(b: *ScopeBuilder, enum_type: *node.EnumType) void {
    b.pop(enum_type.scope.?);
}

pub fn enterFunType(b: *ScopeBuilder, fun_type: *node.FunType) void {
    fun_type.scope = b.push();
}

pub fn exitFunType(b: *ScopeBuilder, fun_type: *node.FunType) void {
    b.pop(fun_type.scope.?);
}

pub fn enterIfStmt(b: *ScopeBuilder, if_stmt: *node.IfStmt) void {
    if_stmt.scope = b.push();
}

pub fn exitIfStmt(b: *ScopeBuilder, if_stmt: *node.IfStmt) void {
    b.pop(if_stmt.scope.?);
}

pub fn enterWhileStmt(b: *ScopeBuilder, while_stmt: *node.WhileStmt) void {
    while_stmt.scope = b.push();
}

pub fn exitWhileStmt(b: *ScopeBuilder, while_stmt: *node.WhileStmt) void {
    b.pop(while_stmt.scope.?);
}

pub fn enterCaseArm(b: *ScopeBuilder, arm: *node.CaseArm) void {
    arm.scope = b.push();
}

pub fn exitCaseArm(b: *ScopeBuilder, arm: *node.CaseArm) void {
    b.pop(arm.scope.?);
}

// -- Attach symbols to scopes --

pub fn enterTypeDecl(b: *ScopeBuilder, type_decl: *node.TypeDecl) void {
    b.insert(type_decl);
}

pub fn enterVarDecl(b: *ScopeBuilder, var_decl: *node.VarDecl) void {
    b.insert(var_decl);
}

pub fn enterFunParam(b: *ScopeBuilder, fun_param: *node.FunParam) void {
    b.insert(fun_param);
}

pub fn enterStructField(b: *ScopeBuilder, struct_field: *node.StructField) void {
    b.insert(struct_field);
}

pub fn enterEnumerator(b: *ScopeBuilder, enumerator: *node.Enumerator) void {
    b.insert(enumerator);
}

pub fn enterCaseBinding(b: *ScopeBuilder, bind: *node.CaseBinding) void {
    b.insert(bind);
}

pub fn enterSumTypeReduce(b: *ScopeBuilder, reduce: *node.SumTypeReduce) void {
    b.insert(reduce);
}

fn push(b: *ScopeBuilder) *node.Scope {
    const scope = b.ast.allocScope();
    b.scopes.append(b.arena.allocator(), scope) catch @panic("OOM");
    return scope;
}

fn pop(b: *ScopeBuilder, scope: *node.Scope) void {
    const result = b.scopes.pop();
    std.debug.assert(result != null and result.? == scope);
}

fn top(b: *ScopeBuilder) *node.Scope {
    return b.scopes.at(b.scopes.len - 1).*;
}

fn insert(b: *ScopeBuilder, symbol: anytype) void {
    if (b.top().insert(b.ctx().allocator, symbol)) |existing| {
        b.code.raise(
            b.ctx().error_out,
            symbol.name.head.position,
            "{s} redeclared in this block; other declaration at {f}",
            .{
                symbol.name.text(),
                b.code.target(existing.head().position),
            },
        ) catch unreachable;
    }
}

fn ctx(b: *ScopeBuilder) *GeneralContext {
    return b.ast.ctx;
}

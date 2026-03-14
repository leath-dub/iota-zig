const std = @import("std");

const Ast = @import("Ast.zig");
const node = @import("node.zig");
const Code = @import("Code.zig");
const GeneralContext = @import("GeneralContext.zig");
const common = @import("common.zig");

const ImperativeScopeResolver = @This();

ast: *Ast,
code: *Code,
arena: std.heap.ArenaAllocator,
global_scope: *node.Scope,
scopes: std.SegmentedList(*node.Scope, 128) = .{},
label_scopes: std.SegmentedList(*node.LabelScope, 128) = .{},

pub fn init(ast: *Ast, code: *Code) ImperativeScopeResolver {
    return .{
        .ast = ast,
        .code = code,
        .arena = ast.ctx.createLifetime(),
        .global_scope = &ast.root.?.scope,
    };
}

pub fn deinit(ir: *ImperativeScopeResolver) void {
    ir.arena.deinit();
}

pub fn enterSourceFile(ir: *ImperativeScopeResolver, source_file: *node.SourceFile) void {
    ir.push(&source_file.scope);
}

pub fn exitSourceFile(ir: *ImperativeScopeResolver, source_file: *node.SourceFile) void {
    ir.pop(&source_file.scope);
}

pub fn enterFunDecl(ir: *ImperativeScopeResolver, fun_decl: *node.FunDecl) void {
    ir.push(&fun_decl.scope);
    ir.pushLabelScope(&fun_decl.label_scope);
}

pub fn exitFunDecl(ir: *ImperativeScopeResolver, fun_decl: *node.FunDecl) void {
    ir.popLabelScope(&fun_decl.label_scope);
    ir.pop(&fun_decl.scope);
}

pub fn enterCompStmt(ir: *ImperativeScopeResolver, comp_stmt: *node.CompStmt) void {
    ir.push(&comp_stmt.scope);
}

pub fn exitCompStmt(ir: *ImperativeScopeResolver, comp_stmt: *node.CompStmt) void {
    ir.pop(&comp_stmt.scope);
}

pub fn exitFunParam(ir: *ImperativeScopeResolver, fun_param: *node.FunParam) void {
    ir.insert(fun_param);
}

// Insert variable declaration on exit so the rhs does not have access to it.
pub fn exitVarDecl(ir: *ImperativeScopeResolver, var_decl: *node.VarDecl) void {
    if (ir.top() != ir.global_scope) {
        ir.insert(var_decl);
    }
}

pub fn enterScopedIdent(ir: *ImperativeScopeResolver, scoped_ident: *node.ScopedIdent) void {
    if (scoped_ident.isGlobal()) {
        if (common.resolveScoped(ir.global_scope, scoped_ident, common.defaultResolveLocal) == null) {
            ir.raise(scoped_ident.head.position, "undefined: {f}", .{scoped_ident});
        }
    } else {
        _ = common.resolveScoped(ir.top(), scoped_ident, common.defaultResolveLocal);
    }
}

// Type names are resolved by PostModuleScopeResolver
pub fn enterType(_: *ImperativeScopeResolver, _: *node.Type) Ast.ChildDisposition {
    return .skip;
}

fn push(ir: *ImperativeScopeResolver, ref: *node.Scope) void {
    ir.scopes.append(ir.arena.allocator(), ref) catch @panic("OOM");
}

fn pop(ir: *ImperativeScopeResolver, scope: *node.Scope) void {
    const result = ir.scopes.pop();
    std.debug.assert(result != null and result.? == scope);
}

fn pushLabelScope(ir: *ImperativeScopeResolver, ref: *node.LabelScope) void {
    ir.label_scopes.append(ir.arena.allocator(), ref) catch @panic("OOM");
}

fn popLabelScope(ir: *ImperativeScopeResolver, scope: *node.LabelScope) void {
    const result = ir.label_scopes.pop();
    std.debug.assert(result != null and result.? == scope);
}

fn top(ir: *ImperativeScopeResolver) *node.Scope {
    return ir.scopes.at(ir.scopes.len - 1).*;
}

fn topOrNull(ir: *ImperativeScopeResolver) ?*node.Scope {
    if (ir.scopes.len == 0) {
        return null;
    }
    return ir.top();
}

fn topLabelScope(ir: *ImperativeScopeResolver) *node.LabelScope {
    return ir.label_scopes.at(ir.label_scopes.len - 1).*;
}

fn ctx(ir: *ImperativeScopeResolver) *GeneralContext {
    return ir.ast.ctx;
}

inline fn raise(ir: *ImperativeScopeResolver, at: Code.Offset, comptime fmt: []const u8, args: anytype) void {
    ir.code.raise(ir.ctx().error_out, at, fmt, args) catch unreachable;
}

fn insert(ir: *ImperativeScopeResolver, symbol: anytype) void {
    if (ir.top().insert(ir.ctx().allocator, symbol)) |existing| {
        ir.raise(
            symbol.name.head.position,
            "{s} redeclared in this block; other declaration at {f}",
            .{
                symbol.name.text(),
                ir.code.target(existing.head().position),
            },
        );
    }
}

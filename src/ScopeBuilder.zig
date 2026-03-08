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
label_scopes: std.SegmentedList(*node.LabelScope, 128) = .{},

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
    b.push(&source_file.scope);
}

pub fn exitSourceFile(b: *ScopeBuilder, source_file: *node.SourceFile) void {
    b.pop(&source_file.scope.?);

    // Resolve function declarations
    const FunDeclResolver = struct {
        ctx: *GeneralContext,
        code: *Code,

        pub fn enterFunDecl(fr: *@This(), fun_decl: *node.FunDecl) void {
            var target_scope = fun_decl.scope.?.parent.?;

            if (fun_decl.type) |type_name| {
                if (target_scope.get(type_name.text())) |ent| {
                    target_scope = switch (ent) {
                        .type_decl => |td| &td.scope.?,
                        else => {
                            fr.code.raise(
                                fr.ctx.error_out,
                                type_name.head.position,
                                "{s} does not resolve to a type",
                                .{
                                    type_name.text(),
                                },
                            ) catch unreachable;
                            return;
                        },
                    };
                }
            }

            if (target_scope.insert(fr.ctx.allocator, fun_decl)) |existing| {
                fr.code.raise(
                    fr.ctx.error_out,
                    fun_decl.name.head.position,
                    "{s} redeclared in this block; other declaration at {f}",
                    .{
                        fun_decl.name.text(),
                        fr.code.target(existing.head().position),
                    },
                ) catch unreachable;
            }
        }
    };
    
    var fr: FunDeclResolver = .{ .ctx = b.ctx(), .code = b.code };
    Ast.walk(&fr, source_file);
}

pub fn enterSumType(b: *ScopeBuilder, sum_type: *node.SumType) void {
    b.push(&sum_type.scope);
}

pub fn exitSumType(b: *ScopeBuilder, sum_type: *node.SumType) void {
    b.pop(&sum_type.scope.?);
}

pub fn enterStructType(b: *ScopeBuilder, struct_type: *node.StructType) void {
    b.push(&struct_type.scope);
}

pub fn exitStructType(b: *ScopeBuilder, struct_type: *node.StructType) void {
    b.pop(&struct_type.scope.?);
}

pub fn enterFunDecl(b: *ScopeBuilder, fun_decl: *node.FunDecl) void {
    b.push(&fun_decl.scope);
    b.pushLabelScope(&fun_decl.label_scope);
}

pub fn exitFunDecl(b: *ScopeBuilder, fun_decl: *node.FunDecl) void {
    b.pop(&fun_decl.scope.?);
    b.popLabelScope(&fun_decl.label_scope.?);
}

pub fn enterCompStmt(b: *ScopeBuilder, comp_stmt: *node.CompStmt) void {
    b.push(&comp_stmt.scope);
}

pub fn exitCompStmt(b: *ScopeBuilder, comp_stmt: *node.CompStmt) void {
    b.pop(&comp_stmt.scope.?);
}

pub fn enterEnumType(b: *ScopeBuilder, enum_type: *node.EnumType) void {
    b.push(&enum_type.scope);
}

pub fn exitEnumType(b: *ScopeBuilder, enum_type: *node.EnumType) void {
    b.pop(&enum_type.scope.?);
}

pub fn enterFunType(b: *ScopeBuilder, fun_type: *node.FunType) void {
    b.push(&fun_type.scope);
}

pub fn exitFunType(b: *ScopeBuilder, fun_type: *node.FunType) void {
    b.pop(&fun_type.scope.?);
}

pub fn enterIfStmt(b: *ScopeBuilder, if_stmt: *node.IfStmt) void {
    b.push(&if_stmt.scope);
}

pub fn exitIfStmt(b: *ScopeBuilder, if_stmt: *node.IfStmt) void {
    b.pop(&if_stmt.scope.?);
}

pub fn enterWhileStmt(b: *ScopeBuilder, while_stmt: *node.WhileStmt) void {
    b.push(&while_stmt.scope);
}

pub fn exitWhileStmt(b: *ScopeBuilder, while_stmt: *node.WhileStmt) void {
    b.pop(&while_stmt.scope.?);
}

pub fn enterCaseArm(b: *ScopeBuilder, arm: *node.CaseArm) void {
    b.push(&arm.scope);
}

pub fn exitCaseArm(b: *ScopeBuilder, arm: *node.CaseArm) void {
    b.pop(&arm.scope.?);
}

pub fn enterTypeDecl(b: *ScopeBuilder, type_decl: *node.TypeDecl) void {
    b.insert(type_decl);
    b.push(&type_decl.scope);
}

pub fn exitTypeDecl(b: *ScopeBuilder, type_decl: *node.TypeDecl) void {
    b.pop(&type_decl.scope.?);
}

// -- Attach symbols to scopes --

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

pub fn enterLabelledStmt(b: *ScopeBuilder, stmt: *node.LabelledStmt) void {
    b.insertLabel(stmt);
}

fn push(b: *ScopeBuilder, ref: *?node.Scope) void {
    ref.* = .{ .parent = b.topOrNull() };
    b.ast.registerSymbolMap(.{ .scope = &ref.*.? });
    b.scopes.append(b.arena.allocator(), &ref.*.?) catch @panic("OOM");
}

fn pop(b: *ScopeBuilder, scope: *node.Scope) void {
    const result = b.scopes.pop();
    std.debug.assert(result != null and result.? == scope);
}

fn pushLabelScope(b: *ScopeBuilder, ref: *?node.LabelScope) void {
    ref.* = .{};
    b.ast.registerSymbolMap(.{ .label_scope = &ref.*.? });
    b.label_scopes.append(b.arena.allocator(), &ref.*.?) catch @panic("OOM");
}

fn popLabelScope(b: *ScopeBuilder, scope: *node.LabelScope) void {
    const result = b.label_scopes.pop();
    std.debug.assert(result != null and result.? == scope);
}

fn top(b: *ScopeBuilder) *node.Scope {
    return b.scopes.at(b.scopes.len - 1).*;
}

fn topOrNull(b: *ScopeBuilder) ?*node.Scope {
    if (b.scopes.len == 0) {
        return null;
    }
    return b.top();
}

fn topLabelScope(b: *ScopeBuilder) *node.LabelScope {
    return b.label_scopes.at(b.label_scopes.len - 1).*;
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

fn insertLabel(b: *ScopeBuilder, symbol: *node.LabelledStmt) void {
    if (b.topLabelScope().insert(b.ctx().allocator, symbol)) |existing| {
        b.code.raise(
            b.ctx().error_out,
            symbol.name.head.position,
            "{s} redeclared in this block; other declaration at {f}",
            .{
                symbol.name.text(),
                b.code.target(existing.head.position),
            },
        ) catch unreachable;
    }
}

fn ctx(b: *ScopeBuilder) *GeneralContext {
    return b.ast.ctx;
}

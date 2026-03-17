const std = @import("std");

const Ast = @import("Ast.zig");
const node = @import("node.zig");
const Code = @import("Code.zig");
const GeneralContext = @import("GeneralContext.zig");
const common = @import("common.zig");

const PostModuleScopeResolver = @This();

ast: *Ast,
code: *Code,
arena: std.heap.ArenaAllocator,
global_scope: *node.Scope,
scopes: std.SegmentedList(*node.Scope, 128) = .{},
label_scopes: std.SegmentedList(*node.LabelScope, 128) = .{},

pub fn init(ast: *Ast, code: *Code) PostModuleScopeResolver {
    return .{
        .ast = ast,
        .code = code,
        .arena = ast.ctx.createLifetime(),
        .global_scope = &ast.root.?.scope,
    };
}

pub fn deinit(pr: *PostModuleScopeResolver) void {
    pr.arena.deinit();
}

pub fn enterSourceFile(pr: *PostModuleScopeResolver, source_file: *node.SourceFile) void {
    pr.push(&source_file.scope);
}

pub fn exitSourceFile(pr: *PostModuleScopeResolver, source_file: *node.SourceFile) void {
    pr.pop(&source_file.scope);

    const CycleDetector = struct {
        pr: *PostModuleScopeResolver,

        pub fn enterTypeDecl(_: *@This(), type_decl: *node.TypeDecl) void {
            type_decl.head.flags.insert(.resolving);
        }

        pub fn exitTypeDecl(cd: *@This(), type_decl: *node.TypeDecl) void {
            var tmp_arena = cd.pr.ctx().createLifetime();
            defer tmp_arena.deinit();

            var rhs = &type_decl.type;
            while (switch (rhs.*) {
                .scoped_ident => |*si| out: {
                    break :out si.resolves_to;
                },
                else => null,
            }) |symbol| {
                switch (symbol.data) {
                    .type_decl => |sub_decl| {
                        if (sub_decl.head.flags.contains(.resolving)) {
                            cd.pr.raise(sub_decl.head.position, "alias cycle detected", .{});
                            break;
                        }
                        rhs = &symbol.data.type_decl.type;
                    },
                    else => break,
                }
            }

            type_decl.head.flags.remove(.resolving);
        }
    };

    var cycle_detector = CycleDetector{ .pr = pr };
    Ast.walk(&cycle_detector, source_file);
}

pub fn enterFunDecl(pr: *PostModuleScopeResolver, fun_decl: *node.FunDecl) void {
    defer {
        pr.push(&fun_decl.scope);
        pr.pushLabelScope(&fun_decl.label_scope);
    }

    var target_scope = pr.top();
    if (fun_decl.type_name) |*type_name| {
        const result = common.resolve(pr.top(), type_name);
        const type_scope = if (result) |symbol| switch (symbol.data) {
            .type_decl => |td| &td.scope,
            else => {
                pr.raise(
                    type_name.head.position,
                    "{s} does not resolve to a type",
                    .{
                        type_name.text(),
                    },
                );
                return;
            },
        } else {
            pr.raise(type_name.head.position, "undefined: {s}", .{type_name.text()});
            return;
        };
        target_scope = type_scope;
    }

    // Should have already been resolved by `ModuleScopeResolver`
    if (target_scope == pr.global_scope) {
        return;
    }

    // TODO: restrict non-local Foo__bar if Foo::bar exists
    // TODO: restrict name of non-local function to not include ' (e.g. func')
    if (target_scope.insert(pr.ctx().allocator, node.Symbol.fromSymbolLike(fun_decl))) |existing| {
        pr.raise(
            fun_decl.name.head.position,
            "{s} redeclared in this block; other declaration at {f}",
            .{
                fun_decl.name.text(),
                pr.code.target(existing.head().position),
            },
        );
    }
}

pub fn enterDefDecl(pr: *PostModuleScopeResolver, def_decl: *node.DefDecl) void {
    var target_scope = pr.top();
    if (def_decl.type_name) |*type_name| {
        const result = common.resolve(pr.top(), type_name);
        const type_scope = if (result) |symbol| switch (symbol.data) {
            .type_decl => |td| &td.scope,
            else => {
                pr.raise(
                    type_name.head.position,
                    "{s} does not resolve to a type",
                    .{
                        type_name.text(),
                    },
                );
                return;
            },
        } else {
            pr.raise(type_name.head.position, "undefined: {s}", .{type_name.text()});
            return;
        };
        target_scope = type_scope;
    }

    // Should have already been resolved by `ModuleScopeResolver`
    if (target_scope == pr.global_scope) {
        return;
    }

    if (target_scope.insert(pr.ctx().allocator, node.Symbol.fromSymbolLike(def_decl))) |existing| {
        pr.raise(
            def_decl.name.head.position,
            "{s} redeclared in this block; other declaration at {f}",
            .{
                def_decl.name.text(),
                pr.code.target(existing.head().position),
            },
        );
    }
}

pub fn exitFunDecl(pr: *PostModuleScopeResolver, fun_decl: *node.FunDecl) void {
    pr.pop(&fun_decl.scope);
    pr.popLabelScope(&fun_decl.label_scope);
}

pub fn enterSumType(pr: *PostModuleScopeResolver, sum_type: *node.SumType) void {
    pr.push(&sum_type.scope);
}

pub fn exitSumType(pr: *PostModuleScopeResolver, sum_type: *node.SumType) void {
    pr.pop(&sum_type.scope);
}

pub fn enterTupleType(pr: *PostModuleScopeResolver, tuple_type: *node.TupleType) void {
    pr.push(&tuple_type.scope);
}

pub fn exitTupleType(pr: *PostModuleScopeResolver, tuple_type: *node.TupleType) void {
    const scope = pr.top();
    for (tuple_type.types, 0..) |*ty, index| {
        std.debug.assert(scope.insert(pr.ctx().allocator, .{
            .name = pr.ast.num(index),
            .data = node.Symbol.Data.fromSymbolLike(ty),
        }) == null);
    }
    pr.pop(&tuple_type.scope);
}

pub fn enterStructType(pr: *PostModuleScopeResolver, struct_type: *node.StructType) void {
    pr.push(&struct_type.scope);
}

pub fn exitStructType(pr: *PostModuleScopeResolver, struct_type: *node.StructType) void {
    pr.pop(&struct_type.scope);
}

pub fn enterCompStmt(pr: *PostModuleScopeResolver, comp_stmt: *node.CompStmt) void {
    pr.push(&comp_stmt.scope);
}

pub fn exitCompStmt(pr: *PostModuleScopeResolver, comp_stmt: *node.CompStmt) void {
    pr.pop(&comp_stmt.scope);
}

pub fn enterEnumType(pr: *PostModuleScopeResolver, enum_type: *node.EnumType) Ast.ChildDisposition {
    pr.push(&enum_type.scope);
    for (enum_type.alts) |*alt| {
        pr.insert(node.Symbol{
            .name = alt.name.text(),
            .data = node.Symbol.Data.fromSymbolLike(alt),
            .type_ctx = .{ .enum_type = enum_type },
        });
    }
    return .skip; // no need to visit children as they are handled above
}

pub fn exitEnumType(pr: *PostModuleScopeResolver, enum_type: *node.EnumType) void {
    pr.pop(&enum_type.scope);
}

pub fn enterTypeDecl(pr: *PostModuleScopeResolver, type_decl: *node.TypeDecl) void {
    if (pr.top() != pr.global_scope) {
        pr.insert(type_decl);
    }
    pr.push(&type_decl.scope);
}

pub fn exitTypeDecl(pr: *PostModuleScopeResolver, type_decl: *node.TypeDecl) void {
    pr.pop(&type_decl.scope);
}

pub fn enterLabelledStmt(pr: *PostModuleScopeResolver, labelled_stmt: *node.LabelledStmt) void {
    pr.insertLabel(labelled_stmt);
}

// Skip any expressions, these are resolved by `ImperativeScopeResvoler`
pub fn enterExpr(_: *PostModuleScopeResolver, _: *node.Expr) Ast.ChildDisposition {
    return .skip;
}

pub fn enterScopedIdent(pr: *PostModuleScopeResolver, scoped_ident: *node.ScopedIdent) void {
    const res = common.resolveScoped(pr.top(), scoped_ident, .{});
    if (res) |r| {
        switch (r.data) {
            .type_decl, .sub_type => {},
            else => {
                pr.raise(scoped_ident.head.position, "{f} is not a type", .{scoped_ident});
            },
        }
    } else {
        pr.raise(scoped_ident.head.position, "undefined: {f}", .{scoped_ident});
    }
}

pub fn enterStructField(pr: *PostModuleScopeResolver, struct_field: *node.StructField) void {
    pr.insert(struct_field);
}

fn push(pr: *PostModuleScopeResolver, ref: *node.Scope) void {
    ref.parent = pr.topOrNull();
    pr.scopes.append(pr.arena.allocator(), ref) catch @panic("OOM");
}

fn pop(pr: *PostModuleScopeResolver, scope: *node.Scope) void {
    const result = pr.scopes.pop();
    std.debug.assert(result != null and result.? == scope);
}

fn pushLabelScope(pr: *PostModuleScopeResolver, ref: *node.LabelScope) void {
    pr.label_scopes.append(pr.arena.allocator(), ref) catch @panic("OOM");
}

fn popLabelScope(pr: *PostModuleScopeResolver, scope: *node.LabelScope) void {
    const result = pr.label_scopes.pop();
    std.debug.assert(result != null and result.? == scope);
}

fn top(pr: *PostModuleScopeResolver) *node.Scope {
    return pr.scopes.at(pr.scopes.len - 1).*;
}

fn topOrNull(pr: *PostModuleScopeResolver) ?*node.Scope {
    if (pr.scopes.len == 0) {
        return null;
    }
    return pr.top();
}

fn topLabelScope(pr: *PostModuleScopeResolver) *node.LabelScope {
    return pr.label_scopes.at(pr.label_scopes.len - 1).*;
}

fn insert(pr: *PostModuleScopeResolver, symbol_: anytype) void {
    const position = if (@TypeOf(symbol_) != node.Symbol)
        symbol_.name.head.position
    else
        symbol_.head().position;
    const symbol = if (@TypeOf(symbol_) != node.Symbol) node.Symbol.fromSymbolLike(symbol_) else symbol_;
    if (pr.top().insert(pr.ctx().allocator, symbol)) |existing| {
        pr.raise(
            position,
            "{s} redeclared in this block; other declaration at {f}",
            .{
                symbol.name,
                pr.code.target(existing.head().position),
            },
        );
    }
}

fn insertLabel(pr: *PostModuleScopeResolver, symbol: *node.LabelledStmt) void {
    if (pr.topLabelScope().insert(pr.ctx().allocator, symbol)) |existing| {
        pr.raise(
            symbol.name.head.position,
            "{s} redeclared in this block; other declaration at {f}",
            .{
                symbol.name.text(),
                pr.code.target(existing.head.position),
            },
        );
    }
}

fn ctx(pr: *PostModuleScopeResolver) *GeneralContext {
    return pr.ast.ctx;
}

inline fn raise(pr: *PostModuleScopeResolver, at: Code.Offset, comptime fmt: []const u8, args: anytype) void {
    pr.code.raise(pr.ctx().error_out, at, fmt, args) catch unreachable;
}

const std = @import("std");

const Ast = @import("Ast.zig");
const node = @import("node.zig");
const Code = @import("Code.zig");
const GeneralContext = @import("GeneralContext.zig");

const IdentResolver = @This();

ast: *Ast,
code: *Code,
arena: std.heap.ArenaAllocator,
scopes: std.SegmentedList(*node.Scope, 128) = .{},
label_scopes: std.SegmentedList(*node.LabelScope, 128) = .{},
// You cannot have a variable declaration as a child of another variable
// declaration so it is safe to have a single variable store this context.
// (i.e. no need for stack)
current_var_decl: ?*node.VarDecl = null,

pub fn init(ast: *Ast, code: *Code) IdentResolver {
    return .{
        .ast = ast,
        .code = code,
        .arena = ast.ctx.createLifetime(),
    };
}

// -- Manage Scopes --

pub fn enter(ir: *IdentResolver, item: Ast.Node) void {
    switch (item) {
        inline else => |foo, tag| {
            const T = @TypeOf(foo.*);
            if (@hasField(T, "scope") and tag != .struct_type) {
                ir.scopes.append(ir.arena.allocator(), &foo.scope.?) catch @panic("OOM");
            }
            if (@hasField(T, "label_scope")) {
                ir.label_scopes.append(ir.arena.allocator(), &foo.label_scope.?) catch @panic("OOM");
            }
        },
    }
}

pub fn exit(ir: *IdentResolver, item: Ast.Node) void {
    switch (item) {
        inline else => |foo, tag| {
            const T = @TypeOf(foo.*);
            if (@hasField(T, "scope") and tag != .struct_type) {
                std.debug.assert(ir.scopes.pop().? == &foo.scope.?);
            }
            if (@hasField(T, "label_scope")) {
                std.debug.assert(ir.label_scopes.pop().? == &foo.label_scope.?);
            }
        },
    }
}

pub fn enterVarDecl(ir: *IdentResolver, var_decl: *node.VarDecl) void {
    std.debug.assert(ir.current_var_decl == null);
    ir.current_var_decl = var_decl;
}

pub fn exitVarDecl(ir: *IdentResolver, var_decl: *node.VarDecl) void {
    std.debug.assert(ir.current_var_decl.? == var_decl);
    ir.current_var_decl = null;
}

// -- Resolve identifiers --
pub fn enterExpr(ir: *IdentResolver, expr: *node.Expr) void {
    if (expr.* == .scoped_ident) {
        const scoped_ident = &expr.scoped_ident;
        if (scoped_ident.isPartial()) {
            return;
        }
        scoped_ident.resolves_to = ir.resolveScoped(scoped_ident);
    }
}

fn resolveLocal(ir: *IdentResolver, scope: *node.Scope, id: *node.Ident) ?node.Symbol {
    const item = scope.get(id.text());
    if (ir.current_var_decl != null and
        item != null and
        item.? == .var_decl and
        item.?.var_decl == ir.current_var_decl.?) {
        return null;
    }
    return item;
}

fn resolve(ir: *IdentResolver, id: *node.Ident) ?node.Symbol {
    var scope = ir.top();
    var item = ir.resolveLocal(scope, id);
    while (item == null and scope.parent != null) {
        scope = scope.parent.?;
        item = ir.resolveLocal(scope, id);
    }
    return item;
}

fn resolveScoped(ir: *IdentResolver, si: *node.ScopedIdent) ?node.Symbol {
    var symbol_opt: ?node.Symbol = null;
    for (si.idents) |*id| {
        var scope_opt: ?*node.Scope = null;
        if (symbol_opt) |symbol| {
            scope_opt = again: switch (symbol) {
                .type_decl => |td| switch (td.type) {
                    .sum => |*sum| &sum.scope.?,
                    .@"enum" => |*en| &en.scope.?,
                    .scoped_ident => |*sub| if (ir.resolveScoped(sub)) |s| continue :again s else null,
                    else => &td.scope.?,
                },
                else => null,
            };
            if (scope_opt == null) {
                symbol_opt = null;
                break;
            }
        }
        symbol_opt = if (scope_opt) |scope|
            ir.resolveLocal(scope, id)
        else ir.resolve(id);
    }
    if (symbol_opt == null) {
        ir.code.raise(
            ir.ctx().error_out,
            si.head.position,
            "undefined: {f}", .{ si } ) catch unreachable;
        return null;
    }
    si.resolves_to = symbol_opt;
    return symbol_opt;
}

fn top(ir: *IdentResolver) *node.Scope {
    return ir.scopes.at(ir.scopes.len - 1).*;
}

fn topLabelScope(ir: *IdentResolver) *node.LabelScope {
    return ir.label_scopes.at(ir.label_scopes.len - 1).*;
}

pub fn deinit(ir: *IdentResolver) void {
    ir.arena.deinit();
}

fn ctx(ir: *IdentResolver) *GeneralContext {
    return ir.ast.ctx;
}

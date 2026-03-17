const std = @import("std");

const Ast = @import("Ast.zig");
const node = @import("node.zig");
const Code = @import("Code.zig");
const GeneralContext = @import("GeneralContext.zig");

const ModuleScopeResolver = @This();

ast: *Ast,
code: *Code,
global_scope: *node.Scope,

pub fn init(ast: *Ast, code: *Code) ModuleScopeResolver {
    return .{
        .ast = ast,
        .code = code,
        .global_scope = &ast.root.?.scope,
    };
}

pub fn enterSourceFile(mr: *ModuleScopeResolver, source_file: *node.SourceFile) void {
    mr.global_scope = &source_file.scope;
}

pub fn enterDecl(mr: *ModuleScopeResolver, decl: *node.Decl) Ast.ChildDisposition {
    switch (decl.*) {
        .@"var" => |*var_decl| {
            mr.insert(var_decl);
        },
        .type => |*type_decl| {
            mr.insert(type_decl);
        },
        .fun => |*fun_decl| {
            if (fun_decl.type_name == null) {
                mr.insert(fun_decl);
            }
        },
        .def => |*def_decl| {
            if (def_decl.type_name == null) {
                mr.insert(def_decl);
            }
        },
        else => {},
    }
    return .skip;
}

fn ctx(mr: *ModuleScopeResolver) *GeneralContext {
    return mr.ast.ctx;
}

fn insert(mr: *ModuleScopeResolver, symbol_: anytype) void {
    const position = if (@TypeOf(symbol_) != node.Symbol)
        symbol_.name.head.position
    else
        symbol_.head().position;
    const symbol = if (@TypeOf(symbol_) != node.Symbol) node.Symbol.fromSymbolLike(symbol_) else symbol_;
    if (mr.global_scope.insert(mr.ctx().allocator, symbol)) |existing| {
        mr.code.raise(
            mr.ctx().error_out,
            position,
            "{s} redeclared in this block; other declaration at {f}",
            .{
                symbol.name,
                mr.code.target(existing.head().position),
            },
        ) catch unreachable;
    }
}

const std = @import("std");
const node = @import("node.zig");

pub fn unqualTypeName(comptime T: type) []const u8 {
    const qualName = @typeName(T);
    if (std.mem.lastIndexOfScalar(u8, qualName, '.')) |index| {
        return qualName[index + 1 ..];
    }
    return qualName;
}

const ResolveFn = fn (scope: *node.Scope, id: *node.Ident) ?node.Symbol;

pub fn defaultResolveLocal(scope: *node.Scope, id: *node.Ident) ?node.Symbol {
    return scope.get(id.text());
}

pub fn resolve(scope_: *node.Scope, id: *node.Ident, comptime resolveLocal: ResolveFn) ?node.Symbol {
    var scope: ?*node.Scope = scope_;
    var item: ?node.Symbol = null;
    while (item == null and scope != null) {
        item = resolveLocal(scope.?, id);
        scope = scope.?.parent;
    }
    return item;
}

pub fn resolveScoped(base: *node.Scope, sid: *node.ScopedIdent, comptime resolveLocal: ResolveFn) ?node.Symbol {
    if (sid.resolves_to) |cache| {
        return cache;
    }

    var symbol_opt: ?node.Symbol = null;
    defer sid.resolves_to = symbol_opt;

    var idents = sid.idents;
    if (sid.isGlobal()) {
        idents = idents[1..];
    }

    for (idents) |*id| {
        if (symbol_opt) |symbol| {
            again: switch (symbol) {
                .type_decl => |td| {
                    // First check in the type scope
                    symbol_opt = resolveLocal(&td.scope, id);

                    // Next check in local sub-scope
                    if (symbol_opt == null) {
                        const fallback_scope = switch (td.type) {
                            .sum => |*sum| &sum.scope,
                            .@"enum" => |*en| &en.scope,
                            .scoped_ident => |*sub| blk: {
                                if (resolveScoped(base, sub, resolveLocal)) |s| {
                                    continue :again s;
                                }
                                break :blk null;
                            },
                            else => null,
                        };
                        if (fallback_scope) |s| {
                            symbol_opt = resolveLocal(s, id);
                        }
                    }
                },
                else => {},
            }
        } else {
            // Lexical lookup if we don't have a currently resolved symbol
            symbol_opt = resolve(base, id, resolveLocal);
        }
    }

    return symbol_opt;
}

const std = @import("std");
const node = @import("node.zig");

pub fn unqualTypeName(comptime T: type) []const u8 {
    const qualName = @typeName(T);
    if (std.mem.lastIndexOfScalar(u8, qualName, '.')) |index| {
        return qualName[index + 1 ..];
    }
    return qualName;
}

pub fn resolveLocal(scope: *const node.Scope, id: *const node.Ident) ?node.Symbol {
    return scope.get(id.text());
}

pub fn resolve(scope_: *const node.Scope, id: *const node.Ident) ?node.Symbol {
    var scope: ?*const node.Scope = scope_;
    var item: ?node.Symbol = null;
    while (item == null and scope != null) {
        item = resolveLocal(scope.?, id);
        scope = scope.?.parent;
    }
    return item;
}

pub const LookupMode = enum {
    local,
    lexical,
};

pub const ResolveConfig = struct {
    symbol: ?node.Symbol = null,
    lookup_mode: LookupMode = .lexical,
};

pub fn resolveScoped(base: *const node.Scope, sid: *node.ScopedIdent, cfg: ResolveConfig) ?node.Symbol {
    if (sid.resolves_to) |cache| {
        return cache;
    }

    var symbol_opt = cfg.symbol;
    defer sid.resolves_to = symbol_opt;

    var idents = sid.idents;
    if (sid.isGlobal()) {
        idents = idents[1..];
    }

    for (idents) |*id| {
        if (symbol_opt) |symbol| {
            again: switch (symbol.data) {
                .type_decl => |td| {
                    // First check in the type scope
                    symbol_opt = resolveLocal(&td.scope, id);

                    // Next check in local sub-scope
                    if (symbol_opt == null) {
                        const fallback_scope = switch (td.type) {
                            .tuple => |*tup| &tup.scope,
                            .sum => |*sum| &sum.scope,
                            .@"enum" => |*en| &en.scope,
                            .scoped_ident => |*sub| blk: {
                                if (resolveScoped(base, sub, cfg)) |s| {
                                    continue :again s.data;
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
            symbol_opt = if (cfg.lookup_mode == .lexical)
                resolve(base, id)
            else resolveLocal(base, id);
        }
    }

    return symbol_opt;
}

pub var index_name_buf: [4096]u8 = undefined;

pub fn indexName(buf: []u8, index: usize) []const u8 {
    return std.fmt.bufPrint(buf, "{d}", .{index}) catch @panic("format error");
}

pub fn todoNoReturn(comptime fmt: []const u8, args: anytype) noreturn {
    std.log.scoped(.todo).debug(fmt, args);
    std.process.exit(1);
}

pub fn todo(cond: bool, comptime fmt: []const u8, args: anytype) void {
    if (cond) return;
    std.log.scoped(.todo).debug(fmt, args);
    std.process.exit(1);
}

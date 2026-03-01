const std = @import("std");
const util = @import("util.zig");
const meta = std.meta;
const assert = std.debug.assert;

const Ast = @import("Ast.zig");
const node = @import("node.zig");

const Lexer = @import("Lexer.zig");
const Token = Lexer.Token;
const TokenType = Lexer.TokenType;
const GeneralContext = @import("GeneralContext.zig");

const Parser = @This();

ctx: *GeneralContext,
ast: Ast,
lexer: Lexer,
panic: bool = false,

pub fn init(ctx: *GeneralContext, lexer: Lexer) Parser {
    return .{
        .ctx = ctx,
        .ast = Ast.init(ctx),
        .lexer = lexer,
    };
}

pub fn parse(p: *Parser) Ast {
    p.ast.root = p.parseSourceFile();
    return p.ast;
}

fn parseSourceFile(p: *Parser) node.SourceFile {
    var sf = p.create(node.SourceFile);

    sf.imports = p.zeroOrMore(node.Import, parseImport, onImportToken);
    sf.decls = p.zeroOrMore(node.Decl, parseDecl, notEof);

    return ok(sf);
}

fn onImportToken(p: *Parser) bool {
    return p.on(.kw_import);
}

fn notEof(p: *Parser) bool {
    return !p.on(.eof);
}

fn parseImport(p: *Parser) node.Import {
    var imp = p.create(node.Import);
    if (!p.skipIf(.kw_import)) return err(imp);
    if (!p.expect(.string_lit)) return err(imp);
    imp.module = p.munch();
    if (!p.skipIf(.semicolon)) return err(imp);
    return ok(imp);
}

fn parseDecl(p: *Parser) node.Decl {
    again: switch (p.at().type) {
        .kw_let, .kw_var, .kw_def => return .{ .@"var" = p.parseVarDecl() },
        // .kw_fun => p.parse_fun_decl(),
        // .kw_type => _ = p.parse_type_decl(.semicolon),
        else => if (p.expectOneOf(.{ .kw_let, .kw_var, .kw_def, .kw_fun, .kw_type })) {
            continue :again p.at().type;
        },
    }
    return .dirty;
}

fn parseVarDecl(p: *Parser) node.VarDecl {
    var var_decl = p.create(node.VarDecl);

    const start = p.munch();
    assert(start.type == .kw_let or start.type == .kw_var or start.type == .kw_def);
    var_decl.declarator = start;

    var_decl.name = p.parseIdent();

    if (p.on(.colon)) {
        _ = p.next();
        var_decl.type = p.parseType();
    }

    if (p.on(.equal)) {
        unreachable;
        // _ = p.next();
        // task.set(p, .init_expr, p.parse_expr());
    }

    if (!p.skipIf(.semicolon)) return err(var_decl);

    return ok(var_decl);
}

fn parseType(p: *Parser) node.Type {
    return again: switch (p.at().type) {
        .kw_s8,
        .kw_u8,
        .kw_s16,
        .kw_u16,
        .kw_s32,
        .kw_u32,
        .kw_s64,
        .kw_u64,
        .kw_f32,
        .kw_f64,
        .kw_bool,
        .kw_string,
        .kw_unit,
        => .{ .builtin = p.createBuiltinType() },
        .lbracket => .{ .coll = p.parseCollType() },
        .kw_struct => .{ .@"struct" = p.parseStructType() },
        // TODO: other types
        else => if (p.expectOneOf(.{ .kw_s8, .kw_u8, .kw_s16, .kw_u16, .kw_s32, .kw_u32, .kw_s64, .kw_u64, .kw_f32, .kw_f64, .kw_bool, .kw_string, .kw_unit, .lbracket, .kw_struct })) {
            continue :again p.at().type;
        } else .dirty,
    };
}

fn createBuiltinType(p: *Parser) node.BuiltinType {
    var builtin_type = p.create(node.BuiltinType);
    builtin_type.token = p.munch();
    return ok(builtin_type);
}

fn parseStructType(p: *Parser) node.StructType {
    var struct_type = p.create(node.StructType);
    if (!p.skipIf(.kw_struct)) return err(struct_type);
    if (!p.skipIf(.lbrace)) return err(struct_type);
    struct_type.fields = p.oneOrMoreCsv(node.StructField, parseStructField, .rbrace);
    if (!p.skipIf(.rbrace)) return err(struct_type);
    return ok(struct_type);
}

fn parseStructField(p: *Parser) node.StructField {
    var field = p.create(node.StructField);
    field.name = p.parseIdent();
    if (!p.skipIf(.colon)) return err(field);
    field.type = p.parseType();
    if (p.on(.equal)) {
        // TODO expressions
        unreachable;
    }
    return ok(field);
}

fn parseCollType(p: *Parser) node.CollType {
    var coll = p.create(node.CollType);
    if (!p.skipIf(.lbracket)) return err(coll);
    if (p.on(.rbracket)) {
        _ = p.next();
        coll.value_type = p.ast.box(p.parseType());
        return ok(coll);
    }
    // TODO expressions
    unreachable;
}

fn parseIdent(p: *Parser) node.Ident {
    var ident = p.create(node.Ident);
    if (p.expect(.ident)) {
        ident.token = p.munch();
        return ok(ident);
    }
    return err(ident);
}

fn raiseExpect(p: *Parser, expected: []const u8) void {
    p.lexer.code.raise(p.ctx.error_out, p.lexer.cursor, "expected {s} got {f}", .{ expected, p.lexer.peek() }) catch unreachable;
}

fn tokenTypes(comptime tts: anytype) [meta.fields(@TypeOf(tts)).len]TokenType {
    const len = meta.fields(@TypeOf(tts)).len;
    var res: [len]TokenType = undefined;
    inline for (0..len) |i| {
        res[i] = @field(tts, std.fmt.comptimePrint("{d}", .{i}));
    }
    return res;
}

fn oneOf(tt: TokenType, comptime tts: []const TokenType) bool {
    inline for (tts) |exp| {
        if (exp == tt) {
            return true;
        }
    }
    return false;
}

fn expect(p: *Parser, comptime tt: TokenType) bool {
    return p.expectOneOf(.{tt});
}

fn expectOneOf(p: *Parser, comptime args: anytype) bool {
    @setEvalBranchQuota(5000);
    const tts = comptime tokenTypes(args);
    const tok = p.lexer.peek();
    const match = oneOf(tok.type, &tts);
    if (!match) {
        // If we are in a panic state already, don't report an error.
        // Try to synchronise on the currently expected token:
        // * if we can't - just backtrack and stay in a panic state.
        // * if we can - leave panic state and return true.
        if (p.panic) {
            const marker = p.lexer;
            p.advance(args);
            const curr = p.lexer.peek();
            if (oneOf(curr.type, &tts)) {
                p.panic = false;
                return true;
            }
            p.lexer = marker;
            return false;
        }
        p.panic = true;
        const FormatToks = struct {
            pub fn format(_: @This(), w: *std.Io.Writer) void {
                if (tts.len == 1) {
                    try w.print("{s}", .{@tagName(tts[0])});
                    return;
                }
                try w.print("one of [", .{});
                inline for (tts, 0..) |tt, i| {
                    if (i != 0) {
                        try w.print(", ", .{});
                    }
                    try w.print("{s}", .{@tagName(tt)});
                }
                try w.print("]", .{});
            }
        };
        p.raiseExpect(std.fmt.comptimePrint("{f}", .{FormatToks{}}));
    } else {
        p.panic = false;
    }
    return match;
}

// "barrier" tokens are tokens which cannot be skipped over when trying to
// syncronize the parser state during panic mode.
fn isBarrierToken(tt: TokenType) bool {
    return switch (tt) {
        .kw_type, .kw_let, .kw_var, .kw_def, .kw_fun => true,
        else => false,
    };
}

fn advance(p: *Parser, comptime args: anytype) void {
    const tts = comptime tokenTypes(args);
    var tok = p.lexer.peek();
    while (tok.type != .eof) : (tok = p.next()) {
        if (isBarrierToken(tok.type)) {
            return;
        }
        inline for (tts) |tt| {
            if (tt == tok.type) {
                return;
            }
        }
    }
}

fn at(p: *Parser) Token {
    return p.lexer.peek();
}

fn next(p: *Parser) Token {
    p.lexer.consume();
    return p.at();
}

fn munch(p: *Parser) Token {
    defer _ = p.next();
    return p.at();
}

fn skipIf(p: *Parser, comptime tt: TokenType) bool {
    if (p.expect(tt)) {
        _ = p.next();
        return true;
    }
    return false;
}

fn on(p: *Parser, tt: TokenType) bool {
    return p.at().type == tt;
}

// Skip token if no progress made by call to `func`
fn ensureProgress(p: *Parser, func: anytype) @typeInfo(@TypeOf(func)).@"fn".return_type.? {
    const cursor = p.lexer.cursor;
    const res = func(p);
    if (cursor == p.lexer.cursor) {
        _ = p.next();
    }
    return res;
}

fn create(p: *Parser, comptime N: type) N {
    var n = N{};
    n.head.position = p.lexer.cursor;
    return n;
}

fn err(n_: anytype) @TypeOf(n_) {
    var n = n_;
    n.head.flags.insert(.dirty);
    return n;
}

fn ok(n_: anytype) @TypeOf(n_) {
    var n = n_;
    if (Ast.lastChild(&n)) |child| switch (child) {
        inline else => |ref| ref.head.flags.insert(.last_child),
    };
    return n;
}

fn zeroOrMore(p: *Parser, comptime T: type, comptime func: fn (p: *Parser) T, comptime cond: fn (p: *Parser) bool) []T {
    var values: std.ArrayList(T) = .{};
    defer values.deinit(p.ctx.allocator);

    while (cond(p)) {
        values.append(p.ctx.allocator, p.ensureProgress(func)) catch @panic("OOM");
    }

    return p.ast.own(T, values.items);
}

fn oneOrMoreCsv(p: *Parser, comptime T: type, comptime func: fn (p: *Parser) T, delim: TokenType) []T {
    var values: std.ArrayList(T) = .{};
    defer values.deinit(p.ctx.allocator);

    values.append(p.ctx.allocator, func(p)) catch @panic("OOM");
    while (p.at().type == .comma) {
        _ = p.next();
        if (p.on(delim)) {
            break;
        }
        values.append(p.ctx.allocator, func(p)) catch @panic("OOM");
    }

    return p.ast.own(T, values.items);
}

const std = @import("std");
const heap = std.heap;
const fs = std.fs;
const log = std.log;
const mem = std.mem;

const Code = @import("Code.zig");
const Lexer = @import("Lexer.zig");
const Ast = @import("Ast.zig");
const Parser = @import("Parser.zig");
const node = @import("node.zig");
const GeneralContext = @import("GeneralContext.zig");
const ty = @import("type.zig");

// Semantic passes
const ModuleScopeResolver = @import("ModuleScopeResolver.zig");
const PostModuleScopeResolver = @import("PostModuleScopeResolver.zig");
const TypeChecker = @import("TypeChecker.zig");
// const ScopeBuilder = @import("ScopeBuilder.zig");
// const IdentResolver = @import("IdentResolver.zig");

const Cli = struct {
    input_path: []const u8,
};

fn parseArgs() error{ParseFailed}!Cli {
    var cli: Cli = undefined;

    var args = std.process.args();
    _ = args.next().?;

    if (args.next()) |file_arg| {
        cli.input_path = file_arg[0..];
    } else {
        log.err("usage: iorac <file to compile>", .{});
        return error.ParseFailed;
    }

    return cli;
}

fn openErrorMsg(e: fs.File.OpenError) ?[]const u8 {
    return switch (e) {
        error.FileNotFound => "file not found",
        error.AccessDenied => "access denied",
        else => null,
    };
}

pub fn main() !void {
    var default_ctx = GeneralContext.Default.init();
    defer default_ctx.deinit();

    var ctx = default_ctx.general();

    const cli = parseArgs() catch std.process.exit(1);
    const input_file = fs.cwd().openFile(cli.input_path, .{}) catch |e| {
        const msg = openErrorMsg(e) orelse @errorName(e);
        log.err("opening file {s}: {s}\n", .{ cli.input_path, msg });
        std.process.exit(1);
    };
    defer input_file.close();

    var cst = ctx.createLifetime();
    defer cst.deinit();

    const text = try input_file.readToEndAlloc(ctx.allocator, std.math.maxInt(usize));
    var code = try Code.init(&cst, cli.input_path, try cst.allocator().dupe(u8, text));
    ctx.allocator.free(text);

    var parser = Parser.init(&ctx, Lexer.init(&ctx, &cst, &code));
    var ast = parser.parse();
    defer ast.deinit();

    if (code.errors != 0) {
        std.debug.print("{f}", .{ast});
        return error.SyntaxAnalysisFailed;
    }

    var msr = ModuleScopeResolver.init(&ast, &code);
    Ast.walk(&msr, &ast.root.?);

    if (code.errors != 0) {
        std.debug.print("{f}", .{ast});
        return error.SemanticAnalysisFailed;
    }

    var tsr = PostModuleScopeResolver.init(&ast, &code);
    Ast.walk(&tsr, &ast.root.?);
    tsr.deinit();

    if (code.errors != 0) {
        std.debug.print("{f}", .{ast});
        return error.SemanticAnalysisFailed;
    }

    var type_store = ty.Store.init(&ctx);
    defer type_store.deinit();

    var tc = TypeChecker.init(&ast, &code, &type_store);
    Ast.walk(&tc, &ast.root.?);
    tc.deinit();

    if (code.errors != 0) {
        std.debug.print("{f}", .{ast});
        return error.SemanticAnalysisFailed;
    }

    std.debug.print("{f}", .{ast});

    // var scope_builder = ScopeBuilder.init(&ast, &code);
    // Ast.walk(&scope_builder, &ast.root.?);
    // scope_builder.deinit();
    //
    // if (code.errors != 0) {
    //     std.debug.print("{f}", .{ast});
    //     return error.SemanticAnalysisFailed;
    // }
    //
    // var ident_resolver = IdentResolver.init(&ast, &code);
    // Ast.walk(&ident_resolver, &ast.root.?);
    // ident_resolver.deinit();
    //
    // if (code.errors != 0) {
    //     std.debug.print("{f}", .{ast});
    //     return error.SemanticAnalysisFailed;
    // }
    //
    // std.debug.print("{f}", .{ast});
}
